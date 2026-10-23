"""Free boundary from the three virtual-casing interface conditions.

:mod:`vmex.core.freeboundary_vc` solves for the boundary on which ``B.n``,
the pressure jump and the sheet current vanish. Lanes: input validation; the
residual rows' definitions on a converged finite-beta state; and (``full``)
the vacuum and finite-beta Landreman--Paul QA cases with their bundled ESSOS
coils, where the solve agrees with NESTOR and is independent of its start.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402
from vmex.core.errors import TrialRejected  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
needs_vc = pytest.mark.skipif(not vc.have_virtual_casing_jax(), reason="requires virtual_casing_jax")


def _deck(name, mpol, ns):
    inp = vj.VmecInput.from_file(DATA / name).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=16)
    return replace(inp, lfreeb=True, mgrid_file="essos_coils(direct)", ns_array=np.array([ns]),
                   niter_array=np.array([5000]), ftol_array=np.array([1e-10]), delt=0.5)


@dataclass(frozen=True)
class _FilamentField:
    """Biot-Savart field of filament coils as a pytree (arrays as leaves, so the model compiles once)."""

    gamma: Any  # (coils, points, 3)
    gamma_dash: Any
    currents: Any  # (coils,)

    def b_cyl(self, r, phi, z):
        xyz = jnp.stack(jnp.broadcast_arrays(r * jnp.cos(phi), r * jnp.sin(phi), z), axis=-1)
        d = xyz[..., None, None, :] - self.gamma
        dB = jnp.cross(self.gamma_dash, d) * (jnp.sum(d * d, axis=-1) ** -1.5)[..., None]
        B = 1e-7 * jnp.mean(jnp.sum(dB * self.currents[:, None, None], axis=-3), axis=-2)
        cos, sin = jnp.cos(phi), jnp.sin(phi)
        return cos * B[..., 0] + sin * B[..., 1], -sin * B[..., 0] + cos * B[..., 1], B[..., 2]


jax.tree_util.register_dataclass(_FilamentField, data_fields=["gamma", "gamma_dash", "currents"], meta_fields=[])


def _coil_field(name):
    """The bundled ESSOS coils as an exact filament field (no mgrid tabulation)."""
    pytest.importorskip("essos")
    from essos.coils import Coils

    coils = Coils.from_json(str(DATA / name))
    return _FilamentField(jnp.asarray(coils.gamma), jnp.asarray(coils.gamma_dash), jnp.asarray(coils.currents))


def test_nonzero_edge_pressure_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), pmass_type="power_series",
                  am=np.array([1.0e3, -0.5e3] + [0.0] * 19))
    with pytest.raises(ValueError, match="edge pressure"):
        fvc.solve_free_boundary_three_term(inp, external_field=lambda xyz: xyz)


def test_asymmetric_boundary_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), lasym=True)
    with pytest.raises(NotImplementedError, match="lasym"):
        fvc.solve_free_boundary_three_term(inp, external_field=lambda xyz: xyz)


def test_unknown_boundary_condition_is_rejected():
    with pytest.raises(ValueError, match="boundary_condition"):
        vj.solve_free_boundary_multigrid(_deck("input.LandremanPaul2021_QA_lowres", 3, 11),
                                         external_field=lambda xyz: xyz, boundary_condition="sheet")


@needs_vc
def test_rows_are_the_three_interface_conditions():
    """Normal row = B_out.n/|B|, sheet rows tangent, and together they are the whole jump."""
    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 11), lfreeb=False)
    eq = opt.solve_equilibrium(inp)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    rows, weights = fvc.boundary_residual(inp, eq.solution, field, runtime=eq.solver_context, nphi=16, ntheta=16)
    data = vc.surface_field_data_from_state(inp, eq.solution, runtime=eq.solver_context, nphi=16, ntheta=16)
    iface = vc.PlasmaVacuumInterface.from_surface_data(data)
    B_in = np.asarray(data.B_total)
    B = np.linalg.norm(B_in, axis=0)
    jump = (np.asarray(iface.total_B_out(field)) - B_in) / B
    normal = np.asarray(iface.normal)
    np.testing.assert_allclose(rows[0], np.asarray(iface.bnormal_residual(field)) / B, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(rows[1], np.asarray(iface.pressure_balance_residual(field)) / (2 * B**2),
                               rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(np.sum(np.asarray(rows[2:]) * normal, axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(rows[0] ** 2 + np.sum(np.asarray(rows[2:]) ** 2, axis=0), np.sum(jump**2, axis=0),
                               rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(np.sum(weights), 1.0, rtol=1e-12)
    summary = fvc.summarize_boundary_residual(rows, weights)
    assert 0 < summary.normal < 0.05 and 0 < summary.sheet_current < 0.05


def test_condensation_rows_vanish_on_a_uniformly_labelled_circle():
    """VMEC's condensation condition holds for a circle in its natural angle, not once that angle is distorted."""
    phi, theta = jnp.meshgrid(jnp.linspace(0.0, jnp.pi / 2, 16, endpoint=False),
                              jnp.linspace(0.0, 2 * jnp.pi, 32, endpoint=False), indexing="ij")

    def rows(angle):
        R = 1.0 + 0.2 * jnp.cos(angle)
        return fvc._condensation_rows(jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), 0.2 * jnp.sin(angle)]), 6, 3)

    assert float(jnp.max(jnp.abs(rows(theta)))) < 1e-12
    assert float(jnp.max(jnp.abs(rows(theta + 0.2 * jnp.sin(2 * theta))))) > 0.1


def test_equal_arc_phase_has_one_branch():
    """The phase rows vanish for the natural angle of a circle and not for that angle shifted by pi (equal arc
    holds for both)."""
    phi, theta = jnp.meshgrid(jnp.linspace(0.0, jnp.pi / 2, 16, endpoint=False),
                              jnp.linspace(0.0, 2 * jnp.pi, 32, endpoint=False), indexing="ij")

    def rows(angle):
        R = 1.0 + 0.2 * jnp.cos(angle)
        return fvc._equal_arc_rows(jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), 0.2 * jnp.sin(angle)]))

    for shift, phase in ((0.0, 0.0), (jnp.pi, jnp.pi)):
        arc, rows_phase = rows(theta + shift)
        assert float(jnp.max(jnp.abs(arc))) < 1e-12
        np.testing.assert_allclose(np.abs(np.asarray(rows_phase)), phase, atol=1e-12)


def test_levenberg_marquardt_finish():
    """The fixed-Jacobian finish: converges on a mildly nonlinear fit, backs off from a failed or worse step,
    and reports why it stopped."""
    target = np.array([1.0, -2.0, 0.5])

    def rows(x):
        return np.r_[x - target, 0.1 * (x[0] * x[1] - target[0] * target[1])]

    def evaluate(x):
        return rows(x), "aux"

    J = np.vstack([np.eye(3), 0.1 * np.array([target[1], target[0], 0.0])])
    x0 = np.zeros(3)
    out = fvc._levenberg_marquardt(evaluate, x0, rows(x0), "aux", J, ftol=1e-12, max_nfev=50, verbose=1)
    assert out["converged"] and out["reason"] == "ftol" and out["accepted"] >= 1
    np.testing.assert_allclose(out["x"], target, atol=1e-5)
    # a step that fails to solve shrinks the radius, a budget of one evaluation stops at once
    failed = fvc._levenberg_marquardt(lambda x: None, x0, rows(x0), "aux", J, ftol=1e-12, max_nfev=30,
                                      verbose=0, radius=10.0)
    assert not failed["converged"] and failed["reason"] in ("radius", "stalled") and failed["accepted"] == 0
    np.testing.assert_array_equal(failed["x"], x0)
    assert fvc._levenberg_marquardt(evaluate, x0, rows(x0), "aux", J, ftol=1e-12, max_nfev=1,
                                    verbose=0)["reason"] == "max_nfev"
    # a wrong Jacobian: steps keep gaining little, so the finish stops as stalled before its budget
    stale = fvc._levenberg_marquardt(evaluate, x0, rows(x0), "aux", np.vstack([np.eye(3), np.ones(3)]) * 40,
                                     ftol=1e-14, max_nfev=200, verbose=0)
    assert stale["reason"] in ("stalled", "radius") and stale["nfev"] < 200


def test_coil_net_current_is_the_circulation_around_the_axis():
    """G of a purely toroidal 1/R field is its R B_phi, whatever the axis loop it is taken around."""

    class Toroidal:
        def b_cyl(self, r, phi, z):
            return 0.0 * r, 0.7 / r, 0.0 * z

    wout = SimpleNamespace(raxis_cc=np.array([1.0, 0.1]), zaxis_cs=np.array([0.0, -0.05]), nfp=2)
    # 720 points and a second-order tangent: 1.2e-5 relative on this axis
    assert fvc._coil_net_current(Toroidal(), wout) == pytest.approx(0.7, rel=1e-4)


#: The smallest vacuum case that still has a free boundary: LP QA at mpol = ntor = 2, ns 7, a 12 x 12 interface
#: grid with the boundary modes up to 1, and its fixed singular quadrature (4 nfp nphi, 2 ntheta).
SMALL = dict(nphi=12, ntheta=12, max_mode=1, quadrature=(96, 24))


def _small_vacuum():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 2, 7), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    return inp, _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")


@needs_vc
@pytest.mark.full
def test_three_term_option_fits_the_best_of_two_starts():
    """The solver option at low resolution: both starts (deck and NESTOR) reduce B.n and the sheet current of
    the deck's fixed-boundary equilibrium, the lower cost wins, and the result reports its interface residual."""
    inp, field = _small_vacuum()
    design = _residual(inp, opt.solve_equilibrium(replace(inp, lfreeb=False)), field, n=12)
    result = vj.solve_free_boundary_multigrid(inp, external_field=field, boundary_condition="three_term",
                                              three_term_options=dict(SMALL, max_nfev=20))
    res = result.boundary_residual
    assert result.vacuum is None and res.normal < design.normal and res.sheet_current < design.sheet_current
    with pytest.raises(ValueError, match="start"):
        vj.solve_free_boundary_multigrid(inp, external_field=field, boundary_condition="three_term",
                                         three_term_options=dict(start="coils"))


@needs_vc
@pytest.mark.full
def test_uncertified_tangent_solve_is_no_jacobian(monkeypatch):
    """A boundary Jacobian whose tangent solves are not all certified raises instead of being used."""
    from vmex.core import implicit as im

    inp, field = _small_vacuum()
    model = fvc.ThreeTermFreeBoundaryModel(replace(inp, lfreeb=False), **SMALL)
    rows, aux = model.evaluate(model.x0, model.params0, field)
    solve = im._implicit_evolved_tangent_multi_rhs

    def one_uncertified(*args, **kwargs):
        solution, report = solve(*args, **kwargs)
        return solution, report._replace(converged=report.converged.at[0].set(False))

    monkeypatch.setattr(im, "_implicit_evolved_tangent_multi_rhs", one_uncertified)
    with pytest.raises(RuntimeError, match="not certified"):
        model.linearize(model.x0, aux, field)


@needs_vc
@pytest.mark.full
def test_three_term_problem_gradient_and_one_slsqp_step():
    """A coil-current design at low resolution: the loss gradient and constraint Jacobian against re-solved central
    differences, one SLSQP step through ``minimize``, and the accepted point's reports."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp, field = _small_vacuum()
    base = field.currents
    problem = FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(1),
        field_from_parameters=lambda x: replace(field, currents=base * (1 + 0.05 * x[0])),
        quantities=(opt.mean_iota,), boundary_condition="three_term", three_term_options=SMALL)
    x0 = problem.x0
    value, grad = problem.value_and_grad(x0)
    jac = problem.constraint_jac(x0)
    h = 1e-3
    fd = [(problem.fun(x0 + h) - problem.fun(x0 - h)) / (2 * h),
          (problem.constraint_values(x0 + h) - problem.constraint_values(x0 - h)) / (2 * h)]
    np.testing.assert_allclose(grad, fd[0], rtol=1e-3)
    np.testing.assert_allclose(jac[:, 0], fd[1], rtol=1e-3)
    result = opt.minimize(problem, x0=x0, method="SLSQP", options=dict(maxiter=1),
                          constraints=[problem.nonlinear_constraint([0.0], [1.0])])
    assert result.accepted_steps == problem.accepted_step == 1 and result.fun <= value
    x = problem.accepted.parameters
    assert max(vars(problem.boundary_residual(x)).values()) < 0.1
    state, params = problem.state(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # the report re-solves the accepted state, not another one
        assert problem.equilibrium_from_x(x).wout.aspect == pytest.approx(result.fun, rel=1e-6)
    problem.close()
    with pytest.raises(ValueError, match="evaluate the candidate"):
        problem.accept_x(x0 + 0.123)
    assert problem.stats["trials"] >= 3 and problem.stats["linearizations"] >= 1


@needs_vc
@pytest.mark.full
def test_boundary_fit_without_an_equilibrium_raises_runtime_error(monkeypatch):
    """A boundary fit whose Jacobian point has no certified equilibrium raises RuntimeError (a rejected trial),
    not KeyError (which ended a campaign run), and never linearizes another point's state."""
    inp, field = _small_vacuum()
    model = fvc.ThreeTermFreeBoundaryModel(replace(inp, lfreeb=False), **SMALL)
    monkeypatch.setattr(model, "evaluate", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="no certified equilibrium"):
        model.solve_boundary(model.params0, field)


def _small_problem(**options):
    """A one-coil-current design on the small vacuum free boundary: aspect ratio, with mean iota as a quantity."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp, field = _small_vacuum()
    base = field.currents
    return FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(1),
        field_from_parameters=lambda x: replace(field, currents=base * (1 + 0.05 * x[0])),
        quantities=(opt.mean_iota,), boundary_condition="three_term", three_term_options=dict(SMALL, **options))


@needs_vc
@pytest.mark.full
def test_failed_or_nonfinite_rows_and_adjoints_reject_the_trial(monkeypatch):
    """A design whose loss or quantity rows raise or are not finite, or whose adjoint fails, is a rejected trial
    (TrialRejected), not an error that ends the optimization and not a usable value or gradient."""
    from vmex.core.errors import AdjointSolveError

    problem = _small_problem()
    rows = problem._rows

    def failing(*args):
        raise RuntimeError("host equilibrium solve failed")

    monkeypatch.setattr(problem, "_rows", failing)
    assert problem._solve(problem.x0 + 0.01) is None
    with pytest.raises(TrialRejected):
        problem.fun(problem.x0 + 0.01)
    monkeypatch.setattr(problem, "_rows", lambda *args: rows(*args) * jnp.nan)
    assert problem._solve(problem.x0 + 0.02) is None
    monkeypatch.setattr(problem, "_rows", rows)

    def failed_adjoint(*args):
        raise AdjointSolveError("adjoint did not converge")

    monkeypatch.setattr(problem, "_pullback", failed_adjoint)
    with pytest.raises(TrialRejected, match="adjoint"):
        problem.value_and_grad(problem.x0 + 0.03)
    problem.close()


def test_levenberg_marquardt_rejects_a_bad_jacobian():
    """A nonfinite boundary Jacobian, or one whose SVD fails, is a RuntimeError (a rejected trial): the
    LinAlgError it used to raise ended a campaign run (QI 2.5%, A, step 240)."""
    evaluate = lambda z: (np.zeros(2), None)  # noqa: E731
    with pytest.raises(RuntimeError, match="nonfinite"):
        fvc._levenberg_marquardt(evaluate, np.zeros(2), np.ones(2), None, np.array([[1.0, np.nan], [0.0, 1.0]]),
                                 ftol=1e-4, max_nfev=5, verbose=0)


def test_levenberg_marquardt_projects_stiff_coordinates():
    """A stiff closure coordinate (x1 in 100 (x1 - x0^2)) stalls the fixed-Jacobian steps of x0; re-solving it
    after every step (project) reaches the minimum in a few evaluations."""
    def evaluate(x):
        return np.array([x[0] - 1.0, 100.0 * (x[1] - x[0] ** 2)]), None

    x0 = np.zeros(2)
    J = np.array([[1.0, 0.0], [0.0, 100.0]])  # at x0
    plain = fvc._levenberg_marquardt(evaluate, x0, evaluate(x0)[0], None, J, ftol=1e-12, max_nfev=8, verbose=0)
    projected = fvc._levenberg_marquardt(evaluate, x0, evaluate(x0)[0], None, J, ftol=1e-12, max_nfev=8, verbose=0,
                                         project=np.array([1]))
    assert projected["converged"] and np.allclose(projected["x"], 1.0, atol=1e-6)
    assert not plain["converged"] and abs(plain["x"][0] - 1.0) > 1e-2


def test_fit_certificate_measures_stationarity_and_rank():
    """The certificate's stationarity is the Gauss-Newton step's predicted relative gain; a repeated column is a
    rank-deficient Jacobian."""
    J = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    assert fvc._fit_certificate(J, np.array([0.0, 0.0, 2.0]))["stationarity"] == pytest.approx(0.0)
    assert fvc._fit_certificate(J, np.array([1.0, 0.0, 1.0]))["stationarity"] == pytest.approx(0.5)
    flat = fvc._fit_certificate(np.diag([1.0, 1e-3, 0.0])[:, :2], np.array([1e-6, 1e-6, 1.0]))
    assert flat["stationarity"] < 1e-11 and flat["step"] == pytest.approx(1e-3)  # flat: tiny gain, long step
    certificate = fvc._fit_certificate(np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]), np.ones(3))
    assert certificate["rank_deficient"] and certificate["condition"] > 1e12


@needs_vc
@pytest.mark.full
def test_unconverged_fits_and_rank_deficient_jacobians_reject_the_trial(monkeypatch):
    """A finite boundary fit that did not converge is no trial point; a rank-deficient boundary Jacobian is no
    gradient; a certified gradient comes with the value of the same (polished) state."""
    problem = _small_problem()
    model = problem.model
    value, gradient = problem.value_and_grad(problem.x0)
    assert problem.fun(problem.x0) == value and problem._record(problem.x0)["certificate"]["stationarity"] <= 1e-3
    solve_boundary = model.solve_boundary
    monkeypatch.setattr(model, "solve_boundary",
                        lambda *args, **kwargs: dict(solve_boundary(*args, **kwargs), converged=False, reason="radius"))
    assert problem._solve(problem.x0 + 0.01) is None and problem.stats["fit_reasons"]["radius"] >= 1
    monkeypatch.setattr(model, "solve_boundary", solve_boundary)
    linearize = model.linearize

    def one_dead_column(*args, **kwargs):
        J, *rest = linearize(*args, **kwargs)
        return (J.at[:, 0].set(0.0) if hasattr(J, "at") else np.where(np.arange(J.shape[1]) == 0, 0.0, J), *rest)

    monkeypatch.setattr(model, "linearize", one_dead_column)
    with pytest.raises(TrialRejected, match="rank-deficient"):
        problem.value_and_grad(problem.x0 + 0.02)
    problem.close()


@needs_vc
@pytest.mark.full
def test_trial_values_come_from_the_tightly_solved_state():
    """With looser trial equilibria (trial_ftol), a trial's value is that of the state its gradient uses: the
    campaign's 1e-11 trials made the plasma-current derivative look 33% off against values from loose states."""
    problem = _small_problem(trial_ftol=1e-6)
    x = problem.x0 + 0.01
    sol = problem._record(x)
    assert sol["aux"][3]
    xj = jnp.asarray(x)
    _, aux = problem.model.evaluate(sol["x"], problem._params_at(xj), problem.field(xj), tight=True)
    tight = np.asarray(problem._rows(aux[0], aux[2], xj))[0]
    assert problem.fun(x) == pytest.approx(float(tight), rel=1e-9)
    problem.close()


@needs_vc
@pytest.mark.full
def test_a_design_with_edge_pressure_is_rejected():
    """The three-term problem needs zero edge pressure: the model refuses a deck with edge pressure, and a design
    whose plasma map gives the edge a pressure is a rejected trial, not a solve that ignores it."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp, field = _small_vacuum()
    with pytest.raises(ValueError, match="edge pressure"):
        fvc.ThreeTermFreeBoundaryModel(replace(inp, lfreeb=False, am=np.r_[1.0e3, np.zeros(20)]), **SMALL)
    problem = FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: field,
        plasma_from_parameters=lambda params, x: replace(params, am=params.am.at[0].add(1.0e3 * x[1])),
        boundary_condition="three_term", three_term_options=SMALL)
    assert problem._solve(np.array([0.0, 0.01])) is None
    with pytest.raises(TrialRejected, match="edge pressure"):
        problem.fun(np.array([0.0, 0.01]))
    problem.close()


def _short_trial_fits(monkeypatch, model):
    """Trial boundary fits that end short of their stationary point (the polish then moves them), as the
    production fits do in soft directions."""
    solve_boundary = model.solve_boundary

    def short(params, field, **kwargs):
        fit = solve_boundary(params, field, **kwargs)
        if not kwargs.get("refresh_on_stall"):  # a polish, not a trial's fit
            return fit
        x = fit["x"] + 1e-3 * np.asarray(model.x_scale)
        rows, aux = model.evaluate(x, params, field)
        return dict(fit, x=x, rows=rows, aux=aux)

    monkeypatch.setattr(model, "solve_boundary", short)


@needs_vc
@pytest.mark.full
def test_a_gradient_never_changes_the_value(monkeypatch):
    """A design has one record: its value, constraint values and boundary are the same before and after its
    gradient is asked for (the gradient used to polish the boundary and overwrite the value already returned)."""
    problem = _small_problem(boundary_ftol=1e-2)
    _short_trial_fits(monkeypatch, problem.model)
    x = problem.x0 + 0.01
    value, constraints = problem.fun(x), problem.constraint_values(x)
    boundary = problem._record(x)["x"].copy()
    assert problem.value_and_grad(x)[0] == value and problem.fun(x) == value
    np.testing.assert_array_equal(problem.constraint_values(x), constraints)
    np.testing.assert_array_equal(problem._record(x)["x"], boundary)
    problem.close()


@needs_vc
@pytest.mark.full
def test_the_residual_check_sees_the_final_state(monkeypatch):
    """boundary_max_residual judges the state the values come from (re-solved tight, polished), not the
    boundary fit's trial state."""
    problem = _small_problem(boundary_ftol=1e-2, trial_ftol=1e-6, boundary_max_residual=1.0)
    model, seen = problem.model, []
    _short_trial_fits(monkeypatch, model)
    residual = model.boundary_residual
    monkeypatch.setattr(model, "boundary_residual", lambda state, *args: seen.append(state) or residual(state, *args))
    sol = problem._record(problem.x0 + 0.01)
    problem._linearize(problem.x0 + 0.01, sol)
    assert seen and seen[-1] is sol["state"]
    problem.close()


@needs_vc
@pytest.mark.full
def test_rejected_trial_leaves_the_continuation(monkeypatch):
    """A rejected trial that changed the boundary fit's trust radius and linearization does not change what the
    next trial starts from."""
    problem = _small_problem()
    model, seen = problem.model, []

    def corrupting_fit(*args, **kwargs):
        x_ref = None if model._linearization is None else np.asarray(model._linearization[0]).copy()
        seen.append((model._radius, x_ref))
        model._radius = 123.0
        if model._linearization is not None:
            model._linearization = (model._linearization[0] + 1.0, *model._linearization[1:])
        raise RuntimeError("no boundary")

    monkeypatch.setattr(model, "solve_boundary", corrupting_fit)
    assert problem._solve(problem.x0 + 0.01) is None and problem._solve(problem.x0 + 0.02) is None
    (radius_1, x_1), (radius_2, x_2) = seen[0], seen[-1]
    assert radius_2 == radius_1 != 123.0
    assert (x_1 is None and x_2 is None) or np.array_equal(x_1, x_2)
    problem.close()


@needs_vc
@pytest.mark.full
def test_plasma_map_derivatives_at_nonzero_offsets():
    """The design gradient differentiates the plasma map itself, phiedge0 (1 + x1/2) + x2^2 / 100: at a nonzero
    offset, not with the updated PHIEDGE as its base, and through x2 although its derivative is zero at x0."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp, field = _small_vacuum()
    base = field.currents
    problem = FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(3),
        field_from_parameters=lambda x: replace(field, currents=base * (1 + 0.05 * x[0])),
        plasma_from_parameters=lambda params, x: replace(
            params, phiedge=params.phiedge * (1 + 0.5 * x[1]) + 0.01 * x[2] ** 2),
        boundary_condition="three_term", three_term_options=SMALL)
    assert list(problem._plasma) == [1, 2]
    x = np.array([0.0, 0.1, 0.3])
    _, grad = problem.value_and_grad(x)
    h = 1e-3
    for k in (1, 2):
        e = np.eye(3)[k] * h
        fd = (problem.fun(x + e) - problem.fun(x - e)) / (2 * h)
        assert grad[k] == pytest.approx(fd, rel=1e-2), k
    problem.close()


@needs_vc
@pytest.mark.full
def test_only_acceptance_advances_the_continuation():
    """A trial that was evaluated and differentiated but not accepted does not change what the next trial
    starts from: x2 after an unaccepted x1 equals x2 straight from the accepted x0."""
    x0 = np.zeros(1)
    seen = []
    for detour in (True, False):
        problem = _small_problem()
        problem.value_and_grad(x0)  # the accepted point's anchor
        if detour:
            problem.value_and_grad(x0 + 0.02)
        seen.append(problem._record(x0 + 0.01)["values"].copy())
        problem.close()
    np.testing.assert_allclose(seen[0], seen[1], rtol=1e-12, atol=0)


@needs_vc
@pytest.mark.full
def test_interior_state_does_not_depend_on_its_start():
    """At one boundary, the equilibrium solved cold, warm from a nearby boundary's equilibrium, and from a
    perturbed seed gives the same interface rows and boundary Jacobian: no history dependence through the
    interior's m = 1 gauge or the hot restart."""
    inp, field = _small_vacuum()
    model = fvc.ThreeTermFreeBoundaryModel(replace(inp, lfreeb=False), **SMALL)
    x = np.asarray(model.x0)
    params = model.params0
    near = model.solve(model.with_boundary(params, jnp.asarray(x + 1e-3 * np.asarray(model.x_scale))))[0]
    perturbed = jax.tree.map(lambda a: a * (1 + 1e-3 * jnp.cos(jnp.arange(a.size).reshape(a.shape)))
                             if jnp.issubdtype(a.dtype, jnp.inexact) else a, model.seed[0])
    from vmex.core import implicit as im

    results = []
    for seed in (None, near, perturbed):
        for cfg in {model.cfg, model.cfg_trial}:  # a genuine cold or seeded solve: no memo, no hot restart
            for cache in (im._LAST_SOLVE, im._HOT_CACHE, im._PERTURB_SEED, im._LAST_REFINED):
                cache.pop(cfg, None)
        model._linearization = None
        rows, aux = model.evaluate(x, params, field, seed=seed, tight=True)
        J = model.linearize(x, aux, field)[0]
        results.append((np.asarray(rows), np.asarray(J)))
        model._last_linearize = None
    for rows, J in results[1:]:
        np.testing.assert_allclose(rows, results[0][0], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(J, results[0][1], rtol=1e-5, atol=1e-7 * np.max(np.abs(results[0][1])))


@needs_vc
@pytest.mark.full
def test_checkpoint_restores_the_accepted_equilibrium(tmp_path):
    """A saved accepted point restores in a new problem of the same definition (the solver's caches cleared),
    reproducing its values; a problem with other settings refuses it."""
    from vmex.core import implicit as im

    problem = _small_problem()
    x1 = problem.x0 + 0.01
    problem.value_and_grad(problem.x0)
    problem.value_and_grad(x1)
    problem.accept_x(x1)
    path = tmp_path / "accepted.npz"
    problem.save_checkpoint(path)
    saved = problem._record(x1)["values"].copy()
    problem.close()
    for cache in (im._LAST_SOLVE, im._HOT_CACHE, im._PERTURB_SEED):
        cache.clear()
    fresh = _small_problem()
    assert fresh.restore_checkpoint(path) <= 1e-6
    assert np.array_equal(fresh.accepted.parameters, x1) and fresh.accepted_step == 1
    np.testing.assert_allclose(fresh._record(x1)["values"], saved, rtol=1e-6)
    # built from the checkpoint: no seed solve at x0, the accepted point is the saved one
    for cache in (im._LAST_SOLVE, im._HOT_CACHE, im._PERTURB_SEED):
        cache.clear()
    resumed = _small_problem(checkpoint=path)
    assert np.array_equal(resumed.accepted.parameters, x1) and resumed.accepted_step == 1
    assert problem.x0.tobytes() not in resumed._cache
    np.testing.assert_allclose(resumed._record(x1)["values"], saved, rtol=1e-6)
    with pytest.raises(ValueError, match="model"):
        _small_problem(label_weight=0.0).restore_checkpoint(path)
    with pytest.raises(ValueError, match="semantic_identity"):  # the caller declares other callbacks
        _small_problem(semantic_identity={"objective": "aspect", "version": 2}).restore_checkpoint(path)
    with pytest.raises(ValueError, match="model"):  # another interior equation set
        _small_problem(m1_complement="frozen").restore_checkpoint(path)
    other, field = _small_problem(), _small_vacuum()[1]  # same settings, another coil-field map
    other.field = lambda x: replace(field, currents=field.currents * (1 + 0.06 * x[0]))
    with pytest.raises(ValueError, match="coil field"):
        other.restore_checkpoint(path)
    # and in a fresh process: nothing but the checkpoint carries the state over
    import subprocess
    import sys

    script = ("import sys; sys.path.insert(0, 'tests'); from test_freeboundary_vc import _small_problem; "
              f"print('change', _small_problem().restore_checkpoint({str(path)!r}))")
    done = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1], capture_output=True,
                          text=True, check=True)
    assert float(done.stdout.split("change")[-1]) <= 1e-6


def _lcfs_distance(w, w0, nphi=5, ntheta=361):
    """Max over sampled planes of the distance from ``w``'s LCFS to ``w0``'s, in metres."""
    theta, worst = np.linspace(0, 2 * np.pi, ntheta), 0.0
    for phi in np.linspace(0, np.pi / int(w.nfp), nphi):
        def rz(x):
            angle = np.outer(theta, np.asarray(x.xm)) - np.asarray(x.xn) * phi
            return np.cos(angle) @ np.asarray(x.rmnc)[-1], np.sin(angle) @ np.asarray(x.zmns)[-1]
        (R, Z), (R0, Z0) = rz(w), rz(w0)
        worst = max(worst, float(np.max(np.min(np.hypot(R[:, None] - R0[None], Z[:, None] - Z0[None]), axis=1))))
    return worst


def _residual(inp, eq, field, n=48):
    return fvc.summarize_boundary_residual(*fvc.boundary_residual(
        inp, eq.solution, field, runtime=eq.solver_context, nphi=n, ntheta=n))


def _nestor(inp, field):
    """NESTOR free boundary as an Equilibrium (fixed-boundary solve on its LCFS) and its deck."""
    free = vj.solve_free_boundary_multigrid(inp, external_field=field, report_boundary_residual=True)
    wout = vj.wout_from_state(inp=inp, state=free.state, fsqr=float(free.fsqr), fsqz=float(free.fsqz),
                              fsql=float(free.fsql), niter=int(free.iterations), converged=bool(free.converged),
                              vacuum_output=free.vacuum)
    rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
    for m, n, r, z in zip(np.asarray(wout.xm, int), np.asarray(wout.xn, int) // int(wout.nfp),
                          np.asarray(wout.rmnc)[-1], np.asarray(wout.zmns)[-1]):
        rbc[n + inp.ntor, m], zbs[n + inp.ntor, m] = r, z
    deck = replace(inp, rbc=rbc, zbs=zbs, lfreeb=False, raxis_c=np.asarray(wout.raxis_cc)[:inp.ntor + 1],
                   zaxis_s=np.asarray(wout.zaxis_cs)[:inp.ntor + 1])
    return opt.solve_equilibrium(deck, initial_state=free.state), deck, free.boundary_residual


@needs_vc
@pytest.mark.full
def test_vacuum_free_boundary_agrees_with_nestor():
    """In vacuum the free boundary is a flux surface of the coil field alone: both methods find it.

    VMEC + NESTOR itself lands ~1.6 mm from the exact free boundary of an
    essentially exact field (VMEC2000 1.3 mm), so the methods are compared at
    that level; from either start the interface conditions end below NESTOR's.
    At mpol = 4 the least-squares minimum is also flat to ~3 mm, so the
    surfaces are compared within the sum of the two floors, and more sharply
    through the residuals and the rotational transform.
    """
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 4, 25), phiedge=-0.025, curtor=0.0,
                  am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    nestor, nestor_deck, reported = _nestor(inp, field)
    before = _residual(inp, nestor, field)
    for name in ("normal", "pressure", "sheet_current"):   # the gate on the NESTOR result
        np.testing.assert_allclose(getattr(reported, name), getattr(before, name), rtol=0.25)
    # The net-current row's reference: outside the plasma the toroidal circulation is the coils' alone, so a
    # free boundary's edge R B_phi equals the coil current that _coil_net_current integrates on the axis.
    np.testing.assert_allclose(abs(fvc._coil_net_current(field, nestor.wout)), abs(float(nestor.wout.rbtor)),
                               rtol=1e-3)
    fits = [fvc.solve_free_boundary_three_term(inp, external_field=field, initial_boundary=start)
            for start in (None, nestor_deck)]
    for fit in fits:
        after = _residual(inp, fit.equilibrium, field)
        assert after.normal < before.normal and after.sheet_current < before.sheet_current
        assert fit.converged and fit.success and "interface residuals" in fit.message
    strict = fvc.solve_free_boundary_three_term(inp, external_field=field, previous=fits[0], max_residual=1e-6)
    assert strict.converged and not strict.success and "above the accepted" in strict.message  # a stated tolerance
    walls = [fit.equilibrium.wout for fit in fits]
    for wall in walls:   # budget: NESTOR's own floor (~1.6 mm) plus the flat least-squares minimum (~3 mm)
        assert _lcfs_distance(wall, nestor.wout) < 5e-3
    np.testing.assert_allclose(np.asarray(walls[0].iotaf)[[0, -1]], np.asarray(nestor.wout.iotaf)[[0, -1]], atol=3e-3)
    # The model's Jacobian (implicit state tangents, field as an argument) is upstream's implicit Jacobian of
    # the same rows at the same state (the seed's: two independent re-solves of a boundary differ at the
    # floor); and a warm restart at an unchanged field stays where it is.
    fit, model = fits[0], fits[0].model
    problem = opt.make_problem(model.fixed, objective_terms=[(lambda state, runtime: model.rows_at(
        state, runtime, field), 0.0, 1.0)], weight_semantics="residual", max_mode=model.max_mode,
        vary_major_radius=True, jacobian_batch_size=8, device=jax.devices()[0])
    J = model.linearize(model.x0, (*model.seed, model.params0, True), field)[0]
    J_upstream = np.asarray(problem.residual_jac(model.x0))
    assert np.linalg.norm(J - J_upstream) < 1e-9 * np.linalg.norm(J_upstream)
    again = fvc.solve_free_boundary_three_term(inp, external_field=field, previous=fit)
    # no new Jacobian: the warm start's own certification reuses the latest linearization (counted once)
    assert again.njev <= 1 and again.converged and again.cost <= fit.cost * (1 + 1e-6)
    assert _lcfs_distance(again.equilibrium.wout, walls[0]) < 1e-4


@needs_vc
@pytest.mark.full
def test_finite_beta_free_boundary_removes_the_sheet_current():
    """LP QA at 0.5% beta with its fitted coils, through the solver's public option."""
    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    design = _residual(inp, opt.solve_equilibrium(replace(inp, lfreeb=False)), field)
    result = vj.solve_free_boundary_multigrid(inp, external_field=field, boundary_condition="three_term")
    assert result.converged and result.vacuum is None
    res = result.boundary_residual
    for name in ("normal", "pressure", "sheet_current"):
        assert getattr(res, name) < 0.25 * getattr(design, name), name
    assert res.sheet_current < 1.5e-3


def test_unknown_problem_boundary_condition_is_rejected():
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    with pytest.raises(ValueError, match="boundary_condition"):
        FreeBoundaryProblem.from_loss(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), lambda s, r, x: 0.0,
                                      np.zeros(1), field_from_parameters=lambda x: None, boundary_condition="sheet")


def test_problem_boundary_condition_dispatches_to_three_term(monkeypatch):
    """``boundary_condition="three_term"`` hands the loss, maps, quantities and options to the three-term problem."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    calls = []
    monkeypatch.setattr(fvc.ThreeTermFreeBoundaryProblem, "from_loss",
                        classmethod(lambda cls, *args, **kwargs: calls.append((args, kwargs)) or "problem"))
    inp, loss, field = _deck("input.LandremanPaul2021_QA_lowres", 3, 11), (lambda s, r, x: 0.0), (lambda x: None)
    got = FreeBoundaryProblem.from_loss(inp, loss, np.zeros(1), field_from_parameters=field, quantities=(len,),
                                        boundary_condition="three_term", three_term_options=dict(nphi=16))
    (args, kwargs), = calls
    assert got == "problem" and args[:2] == (inp, loss) and kwargs["field_from_parameters"] is field
    assert kwargs["quantities"] == (len,) and kwargs["nphi"] == 16


@needs_vc
@pytest.mark.full
def test_three_term_problem_gradients_match_resolved_differences():
    """Design gradients by the implicit function theorem of the boundary fit, against re-solved central
    differences, along a plasma coordinate (PHIEDGE) and a field coordinate (the coil currents)."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 13), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    problem = FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: replace(coils, currents=coils.currents * (1 + x[1])),
        plasma_from_parameters=lambda params, x: replace(params, phiedge=params.phiedge * (1 + x[0])),
        quantities=(opt.major_radius,),
        parameter_quantities=(lambda state, runtime, x: opt.aspect_ratio(state, runtime) * (1 + x[1]),),
        boundary_condition="three_term",
        three_term_options=dict(nphi=24, ntheta=24, quadrature=(4 * 2 * 24, 48), boundary_ftol=1e-10,
                                boundary_max_nfev=80))
    assert isinstance(problem, fvc.ThreeTermFreeBoundaryProblem)
    x0 = problem.x0
    value, grad = problem.value_and_grad(x0)
    jac = problem.constraint_jac(x0)
    assert problem.boundary_residual(x0).sheet_current < 1e-2
    for k, h in ((0, 2e-3), (1, 2e-3)):
        e = h * np.eye(2)[k]
        values = [np.r_[problem.fun(x0 + s * e), problem.constraint_values(x0 + s * e)] for s in (1, -1)]
        fd = (values[0] - values[1]) / (2 * h)
        np.testing.assert_allclose(np.r_[grad[k], jac[:, k]], fd, rtol=2e-2, atol=1e-8 * max(1.0, abs(value)))
    problem.accept_x(x0 + 2e-3 * np.eye(2)[1])
    assert problem.accepted_step == 1
    problem.boundary_max_residual = 1e-12  # no fit gets there: the trial is no solution
    with pytest.raises(TrialRejected):
        problem.fun(x0 + 1e-3 * np.eye(2)[0])


def _lp_beta0p5_profiles(inp):
    """Landreman--Buller--Drevlak kinetic profiles whose 2 e n T is the LP QA 0.5% deck's pressure."""
    from vmex.core import bootstrap as bs

    n0 = 1.5e20
    T0 = float(inp.am[0]) / (2 * bs.ELEMENTARY_CHARGE * n0)
    return bs.KineticProfiles(ne_coeffs=n0 * np.array([1, 0, 0, 0, 0, -1.0]), Te_coeffs=T0 * np.array([1, -1.0]),
                              Ti_coeffs=T0 * np.array([1, -1.0]))


def test_bootstrap_current_profile_layout():
    """A knot per half-mesh surface, I'(0) = 0 and three unknown knots inside s_1, CURTOR its integral."""
    from vmex.core.profiles import current

    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 25), lfreeb=False)
    from vmex.core.bootstrap import HalfMeshCurrent

    block = HalfMeshCurrent(inp, _lp_beta0p5_profiles(inp), 0)
    s = (np.arange(1, 25) - 0.5) / 24
    np.testing.assert_allclose(block.knots[4:-1], s)
    values = np.asarray(block.values(block.x0))
    assert values[0] == 0.0 and block.deck.pcurr_type == "line_segment_ip"
    np.testing.assert_array_equal(block.deck.ac_aux_s, block.knots)
    np.testing.assert_allclose(block.deck.ac_aux_f, values)
    assert block.x0.size == 3 + 24
    assert block.deck.curtor == pytest.approx(float(current("line_segment_ip", inp.ac, block.knots, values, 1.0)))
    # the deck's own enclosed current away from the axis, where only the first cell differs
    deck_I = lambda d, x: float(d.curtor) * np.asarray(current(d.pcurr_type, d.ac, d.ac_aux_s, d.ac_aux_f, x)) / float(  # noqa: E731
        current(d.pcurr_type, d.ac, d.ac_aux_s, d.ac_aux_f, 1.0))
    outer = s[s > 0.2]
    np.testing.assert_allclose(deck_I(block.deck, outer), deck_I(inp, outer), rtol=2e-2)
    with pytest.raises(ValueError, match="ncurr = 1"):
        HalfMeshCurrent(replace(inp, ncurr=0), _lp_beta0p5_profiles(inp), 0)
    with pytest.raises(ValueError, match="knot per half-mesh surface"):
        HalfMeshCurrent(replace(inp, ns_array=np.array([201])), _lp_beta0p5_profiles(inp), 0)
    from vmex.core.bootstrap import self_consistent_bootstrap

    with pytest.raises(ValueError, match="'power_series' or 'half_mesh'"):
        self_consistent_bootstrap(inp, _lp_beta0p5_profiles(inp), 0, profile="spline")


@needs_vc
@pytest.mark.full
def test_bootstrap_current_solved_with_the_free_boundary():
    """LP QA at 0.5% beta: boundary and Redl current in one solve, with exact columns for both."""
    from vmex.core import bootstrap as bs

    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    profiles = _lp_beta0p5_profiles(inp)
    fit = fvc.solve_free_boundary_three_term(inp, external_field=field, bootstrap=profiles, jacobian_ftol=None)
    model = fit.model
    state, mask, params, _ = fit.aux
    assert fit.x.size == model.n_boundary + 3 + 24
    assert model.bootstrap_residual(state, params) < 1e-2
    assert fit.boundary_residual.sheet_current < 2e-3
    # independently of the inversion: VMEC's <J.B> (finite-difference identity) is Redl's away from the ends
    runtime = model._im.runtime_from_params(params, model.cfg)
    hm = bs._half_mesh_fields(state, runtime)
    s = np.asarray(hm.s_half)
    jr = np.asarray(bs.j_dot_B_redl(profiles, bs._geometry_from_half(hm, hm.s_half, n_lambda=bs.N_LAMBDA), 0)[0])
    jv = np.asarray(bs._jv_from_half(hm, hm.s_half))
    inner = (s > 0.1) & (s < 0.9)
    assert np.max(np.abs(jv - jr)[inner]) < 0.05 * np.max(np.abs(jr))
    # Jacobian columns of a boundary and two current coordinates against central differences of re-solved rows
    # (small steps: near the axis Redl's rows are strongly nonlinear in the boundary, through f_t ~ eps^(1/2))
    J = model.linearize(fit.x, fit.aux, field)[0]
    for k in (0, model.n_boundary + 3, model.n_boundary + 15):
        h = 1e-5 * model.x_scale[k]
        rows = [model.evaluate(fit.x + sign * h * np.eye(fit.x.size)[k], params, field, tight=True)[0]
                for sign in (1, -1)]
        fd = (rows[0] - rows[1]) / (2 * h)
        assert np.linalg.norm(J[:, k] - fd) < 1e-4 * np.linalg.norm(fd), k


def test_fixed_boundary_bootstrap_model_matches_picard():
    """fixed_boundary=True: the Gauss-Newton current at a fixed boundary is the Picard loop's, with exact columns."""
    from vmex.core import bootstrap as bs

    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 25), lfreeb=False,
                  ftol_array=np.array([1e-13]), niter_array=np.array([20000]))
    profiles = _lp_beta0p5_profiles(inp)
    model = fvc.ThreeTermFreeBoundaryModel(inp, bootstrap=profiles, fixed_boundary=True)
    assert model.n_boundary == 0 and model.x0.size == 3 + 24  # three knots inside s_1, then the half mesh
    out = model.solve_boundary(model.params0, None, ftol=1e-10)
    state, _, params, _ = out["aux"]
    assert model.bootstrap_residual(state, params) < 1e-4
    picard = bs.self_consistent_bootstrap(inp, profiles, 0, n_iter=40, tol=1e-6, relax=0.5, profile="half_mesh")
    np.testing.assert_allclose(np.asarray(params.ac_aux_f), picard.input.ac_aux_f,
                               rtol=0, atol=2e-4 * np.max(np.abs(picard.input.ac_aux_f)))
    J = model.linearize(out["x"], out["aux"], None)[0]
    k, h = 5, 1e-5 * model.x_scale[5]
    rows = [model.evaluate(out["x"] + sign * h * np.eye(out["x"].size)[k], params, None, tight=True)[0]
            for sign in (1, -1)]
    fd = (rows[0] - rows[1]) / (2 * h)
    assert np.linalg.norm(J[:, k] - fd) < 1e-5 * np.linalg.norm(fd)


CTH_MGRID = DATA / "mgrid_cth_like.nc"
CTH_VMEC2000 = Path(__file__).parent / "data" / "cth_like_vmec2000.json"


def _wout(inp, result):
    return vj.wout_from_state(inp=inp, state=result.state, fsqr=float(result.fsqr), fsqz=float(result.fsqz),
                              fsql=float(result.fsql), converged=bool(result.converged), vacuum_output=result.vacuum)


def _vacuum_flux_surface(w, field, *, lines=16, transits=20, steps=72):
    """How well ``w``'s LCFS is a flux surface of the vacuum ``field``: the RMS coil B.n/|B| on it and the RMS
    distance (m) from it of the phi = 0 crossings of field lines started on it (RK4 in phi)."""
    nfp, xm, xn = int(w.nfp), np.asarray(w.xm, float), np.asarray(w.xn, float)
    rmnc, zmns = np.asarray(w.rmnc)[-1], np.asarray(w.zmns)[-1]
    th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 96, endpoint=False),
                         np.linspace(0, 2 * np.pi / nfp, 36, endpoint=False), indexing="ij")
    c, s_ = np.cos(th[..., None] * xm - ph[..., None] * xn), np.sin(th[..., None] * xm - ph[..., None] * xn)
    R, Z = c @ rmnc, s_ @ zmns
    normal = np.cross(np.stack([(s_ * xn) @ rmnc, R, (-c * xn) @ zmns]),
                      np.stack([(-s_ * xm) @ rmnc, 0 * R, (c * xm) @ zmns]), axis=0)
    B = np.stack([np.asarray(b) for b in field.b_cyl(jnp.asarray(R), jnp.asarray(ph), jnp.asarray(Z))])
    bn = np.sum(B * normal, 0) / np.linalg.norm(B, axis=0) / np.linalg.norm(normal, axis=0)
    h = 2 * np.pi / (nfp * steps)

    def rhs(phi, y):
        BR, Bp, BZ = field.b_cyl(y[0], jnp.full_like(y[0], phi), y[1])
        return jnp.stack([y[0] * BR / Bp, y[0] * BZ / Bp])

    def step(y, phi):
        k1 = rhs(phi, y)
        k2 = rhs(phi + h / 2, y + h / 2 * k1)
        k3 = rhs(phi + h / 2, y + h / 2 * k2)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + rhs(phi + h, y + h * k3))

    period = jax.jit(lambda y: jax.lax.fori_loop(0, steps, lambda k, y: step(y, k * h), y))
    theta = np.linspace(0, 2 * np.pi, lines, endpoint=False)
    y = jnp.asarray([np.cos(np.outer(theta, xm)) @ rmnc, np.sin(np.outer(theta, xm)) @ zmns])
    crossings = []
    for _ in range(transits * nfp):  # the field is nfp-periodic: every period ends on phi = 0
        y = period(y)
        crossings.append(np.asarray(y))
    points = np.concatenate(crossings, axis=1)
    dense = np.linspace(0, 2 * np.pi, 2881)
    curve = np.stack([np.cos(np.outer(dense, xm)) @ rmnc, np.sin(np.outer(dense, xm)) @ zmns], -1)
    distance = np.min(np.linalg.norm(points.T[:, None, :] - curve[None], axis=-1), axis=1)
    return float(np.sqrt(np.mean(bn**2))), float(np.sqrt(np.mean(distance**2)))


@needs_vc
@pytest.mark.full
@pytest.mark.skipif(not CTH_MGRID.exists(), reason="python tools/fetch_assets.py --bundle reference-nc")
@pytest.mark.parametrize("case", ["vacuum", "beta"])
def test_three_term_on_the_cth_like_deck(case):
    """uwplasma/vmex#451: the three-term free boundary of the CTH-like deck, judged by its own quality, with
    VMEC2000's (tests/data/cth_like_vmec2000.json: xvmec on the same deck and mgrid, vacuum with
    PRES_SCALE = CURTOR = 0) scored the same way.

    Vacuum: the coil field is the whole field, so the LCFS must be one of its flux surfaces: the coil B.n/|B| on it
    and the drift of field lines started on it measure that without virtual
    casing.  From the deck boundary alone the fit ends in a least-squares minimum ~4 cm away; the default start (the
    lower cost of the deck and NESTOR starts, on the tricubic mgrid field) lands 2.1 mm from VMEC2000's boundary;
    in that field its coil B.n is 3.0e-4 against VMEC2000's 4.6e-4 and its interface residuals a fifth of
    VMEC2000's, while field lines drift 0.60 mm from it against 0.45 mm (RMS, 20 transits): at 5 x 4 modes neither
    boundary is a flux surface to better than ~0.5 mm, and the least squares weighs every B.n harmonic alike where
    field lines amplify the near-resonant ones.  At 9 x 8 modes and ns 51 the three-term boundary is the better one
    on both counts (coil B.n 1.7e-5 against 3.6e-5, drift 0.038 against 0.046 mm), 0.3 mm from NESTOR's.  Finite beta: field lines need the plasma field too, so the check is the interface residuals:
    VMEC2000's free boundary keeps an edge sheet current (K = 2.4e-3 |B| at 5 x 4 modes, as large up to 9 x 8 and
    ns 31), which the three-term boundary halves.  About 7 minutes per case on one GPU."""
    from vmex.core.mgrid import MgridField

    reference = json.loads(CTH_VMEC2000.read_text())[case]
    inp = replace(vj.VmecInput.from_file(DATA / "input.cth_like_free_bdy"), mgrid_file=str(CTH_MGRID))
    if case == "vacuum":
        inp = replace(inp, pres_scale=0.0, curtor=0.0)
    three = vj.solve_free_boundary_multigrid(inp, boundary_condition="three_term")
    assert three.converged
    w = _wout(inp, three)
    vmec2000 = SimpleNamespace(nfp=int(w.nfp), xm=np.asarray(reference["xm"]), xn=np.asarray(reference["xn"]),
                               rmnc=np.asarray([reference["rmnc"]]), zmns=np.asarray([reference["zmns"]]))
    print(f"{case}: three-term LCFS {1e3 * _lcfs_distance(w, vmec2000):.2f} mm from VMEC2000's; residuals "
          f"{three.boundary_residual} against VMEC2000's {reference['residual']}")
    for name in ("normal", "sheet_current"):
        assert getattr(three.boundary_residual, name) < reference["residual"][name]
    if case == "vacuum":
        field = MgridField.from_input(inp, CTH_MGRID, order=3)  # the smooth field the three-term solve uses
        (bn, drift), (bn_ref, drift_ref) = _vacuum_flux_surface(w, field), _vacuum_flux_surface(vmec2000, field)
        print(f"vacuum: coil B.n {bn:.2e} (VMEC2000 {bn_ref:.2e}), field-line drift {1e3 * drift:.2f} mm "
              f"(VMEC2000 {1e3 * drift_ref:.2f} mm)")
        assert bn < 5e-4 and bn < bn_ref
        assert drift < 1e-3  # absolute: at 5 x 4 modes VMEC2000's boundary drifts a little less (see above)


def _exact_and_gauss_newton_sensitivity(residual, b, d):
    """``db/dd`` of ``min_b |residual(b, d)|^2`` at a stationary ``b``: exact (curvature terms by AD) and
    Gauss-Newton, both through :func:`fvc._least_squares_sensitivity`."""
    r = np.asarray(residual(b, d))
    J_b = np.asarray(jax.jacfwd(residual, 0)(b, d))
    J_d = np.asarray(jax.jacfwd(residual, 1)(b, d))
    S_bb = np.einsum("i,ijk->jk", r, np.asarray(jax.jacfwd(jax.jacfwd(residual, 0), 0)(b, d)))
    S_bd = np.einsum("i,ijk->jk", r, np.asarray(jax.jacfwd(jax.jacfwd(residual, 0), 1)(b, d)))
    return (fvc._least_squares_sensitivity(J_b, J_d, S_bb, S_bd), fvc._least_squares_sensitivity(J_b, J_d),
            J_b, J_d)


@pytest.mark.parametrize("which", ["state_curvature", "mixed"])
def test_least_squares_sensitivity_needs_the_curvature_terms(which):
    """Two analytic least squares at b = 0, d = 0 where the residual does not vanish: exact db/dd = 1.25, Gauss-Newton
    1.  r = (b - d, sqrt(0.1)(b^2 - 1)) needs S_bb; r = (b - d, (b d - 1)/2) needs only the mixed S_bd.  The exact
    sensitivity matches central differences of the re-solved minimum; the gradient's multipliers are Gauss-Newton."""
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
    lam = fvc._residual_multipliers(J_b, G_b)
    assert float(-(lam.T @ J_d)[0, 0]) == pytest.approx(float((G_b @ gauss_newton)[0, 0]), rel=1e-12)


def test_residual_multipliers_equal_the_forward_gauss_newton_sensitivity():
    """The adjoint form of the gradient (multipliers, one per row) equals the forward Gauss-Newton sensitivity."""
    rng = np.random.default_rng(0)
    J_b, J_d, G_b = rng.normal(size=(30, 7)), rng.normal(size=(30, 4)), rng.normal(size=(3, 7))
    lam = fvc._residual_multipliers(J_b, G_b)
    np.testing.assert_allclose(-lam.T @ J_d, G_b @ fvc._least_squares_sensitivity(J_b, J_d), rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Reduced (variable-projection) Levenberg-Marquardt and least-squares sensitivities (PR #121 review)
# ---------------------------------------------------------------------------


def _stiff_closure(weight=100.0):
    """r(b, c) = (b - 1, b - 1/2, weight (c - b^2)): c a stiff closure of b, nonlinear in b."""
    def evaluate(x):
        b, c = x
        return np.array([b - 1.0, b - 0.5, weight * (c - b * b)]), None

    def jacobian(x):
        b, _ = x
        return np.array([[1.0, 0.0], [1.0, 0.0], [-2 * weight * b, weight]])

    return evaluate, jacobian


def test_projected_levenberg_marquardt_predicts_the_composite_step():
    """With the closure eliminated, the model is that of the evaluated composite step (boundary step, closure
    re-solved): the step from a stale Jacobian converges to the minimum, the gain ratio of the first step is near
    one, and the closure is conditionally stationary at every accepted point.  Holding the closure in the joint
    model instead stalls on the curvature of its rows."""
    evaluate, jacobian = _stiff_closure()
    x0 = np.array([3.0, 9.0])  # closure-consistent start, far from the minimum b = 3/4
    r0, _ = evaluate(x0)
    log = []

    def logged(x):
        out = evaluate(x)
        log.append((np.array(x), 0.5 * out[0] @ out[0]))
        return out

    out = fvc._levenberg_marquardt(logged, x0, r0, None, jacobian(x0), ftol=1e-12, max_nfev=40, verbose=0,
                                   project=np.array([1]))
    assert out["converged"] and out["reason"] == "ftol"
    np.testing.assert_allclose(out["x"], [0.75, 0.5625], rtol=1e-6)
    assert out["projections"] == 0
    joint = fvc._levenberg_marquardt(evaluate, x0, r0, None, jacobian(x0), ftol=1e-12, max_nfev=40, verbose=0)
    assert not joint["converged"] and abs(joint["x"][0] - 0.75) > 1e-2


def test_project_stops_at_conditional_stationarity():
    evaluate, jacobian = _stiff_closure()
    x = np.array([0.75, 0.0])
    Q, R = np.linalg.qr(jacobian(x)[:, [1]])
    xs, got, used, stationary = fvc._project(evaluate, x, evaluate(x), Q, R, np.array([1]), 10, 1e-20)
    assert stationary and used == 1  # linear in c: one correction
    np.testing.assert_allclose(xs, [0.75, 0.5625], rtol=1e-14)
    _, _, used, stationary = fvc._project(evaluate, xs, got, Q, R, np.array([1]), 10, 1e-20)
    assert stationary and used == 0


@pytest.mark.parametrize("residual", [
    lambda b, d: jnp.stack([b - d, jnp.sqrt(0.1) * (b * b - 1.0)]),  # state curvature
    lambda b, d: jnp.stack([b - d, 0.5 * (b * d - 1.0)]),  # mixed term only
])
def test_least_squares_sensitivity_exact_vs_gauss_newton(residual):
    """At the stationary point b = 0 of d = 0, exact db/dd = 1.25; Gauss-Newton (no second-order terms) gives
    1, the error the problem's gradient carries at a nonzero residual.  The exact value matches re-solved finite
    differences."""
    import jax
    import scipy.optimize

    b, d = 0.0, 0.0
    r = np.asarray(residual(b, d))
    J_y = np.asarray(jax.jacfwd(residual, 0)(b, d))[:, None]
    J_d = np.asarray(jax.jacfwd(residual, 1)(b, d))
    hess = jax.hessian(residual, (0, 1))(b, d)  # ((d2/db2, d2/db dd), ...), each per residual component
    S_yy, S_yd = r @ np.asarray(hess[0][0]), r @ np.asarray(hess[0][1])
    assert np.asarray(jax.grad(lambda y: 0.5 * jnp.sum(residual(y, d) ** 2))(b)) == 0.0
    exact = fvc.least_squares_sensitivity(J_y, J_d, S_yy, S_yd)
    np.testing.assert_allclose(exact, [[1.25]], rtol=1e-13)
    np.testing.assert_allclose(fvc.least_squares_sensitivity(J_y, J_d), [[1.0]], rtol=1e-13)
    h = 1e-5
    stationarity = jax.grad(lambda y, dd: 0.5 * jnp.sum(residual(y, dd) ** 2))
    solved = [scipy.optimize.brentq(lambda y, dd=dd: float(stationarity(y, dd)), -0.1, 0.1, xtol=1e-15)
              for dd in (-h, h)]
    assert (solved[1] - solved[0]) / (2 * h) == pytest.approx(1.25, rel=1e-6)


@needs_vc
@pytest.mark.full
def test_standalone_and_optimization_accept_the_same_equilibrium(monkeypatch):
    """One definition of solved: at the same deck, field and fit settings the standalone solve and the
    optimization's seed record are both certified by the same rule and land on the same stationary point (within
    their Gauss-Newton steps); a finish that only stalled, far from it, is not converged."""
    inp, field = _small_vacuum()
    problem = _small_problem(boundary_ftol=1e-4, boundary_max_nfev=60)
    record = problem._record(problem.x0)
    fit = fvc.solve_free_boundary_three_term(inp, external_field=field, ftol=1e-4, max_nfev=60, **SMALL)
    assert fit.converged and fit.certificate is not None and "certified" in fit.message
    for certificate in (fit.certificate, record["certificate"]):
        assert certificate["stationarity"] <= 1e-4
    scaled = np.linalg.norm((fit.x - record["x"]) / problem.model.x_scale)
    assert scaled <= fit.certificate["step"] + record["certificate"]["step"] + 1e-8

    def stall(evaluate, x, r, aux, J, **kwargs):
        return dict(x=x, rows=r, aux=aux, nfev=1, accepted=0, converged=False, radius=kwargs.get("radius"),
                    reason="stalled", projections=0)

    monkeypatch.setattr(fvc, "_levenberg_marquardt", stall)
    stalled = fvc.solve_free_boundary_three_term(inp, external_field=field, ftol=1e-4, max_nfev=2, **SMALL)
    assert not stalled.converged and not stalled.success and "not certified" in stalled.message
    assert stalled.certificate is None


@needs_vc
@pytest.mark.full
def test_certify_polishes_on_tight_states(monkeypatch):
    """The polish of a trial that failed its certificate evaluates tight states (the ones the certificate judges),
    while the trial's own fit stays on trial states."""
    problem = _small_problem()
    model, calls, original, real = problem.model, [], problem.model.solve_boundary, fvc._fit_certificate
    count = {"n": 0}

    def spy(*args, **kwargs):
        calls.append(bool(kwargs.get("tight", False)))
        return original(*args, **kwargs)

    def first_fails(J, r, *args):
        count["n"] += 1
        certificate = real(J, r, *args)
        return dict(certificate, stationarity=1.0) if count["n"] == 1 else certificate

    monkeypatch.setattr(model, "solve_boundary", spy)
    monkeypatch.setattr(fvc, "_fit_certificate", first_fails)
    assert problem._solve(problem.x0 + 0.01) is not None
    assert calls[0] is False and True in calls[1:]
    problem.close()


@needs_vc
@pytest.mark.full
def test_normal_operator_is_the_surface_normal_displacement():
    """N @ dx is the normal displacement of the boundary surface for a coordinate change dx (finite differences of
    the surface against its numerical normal, periodic central differences), a the RMS minor radius."""
    inp, field = _small_vacuum()
    model = fvc.ThreeTermFreeBoundaryModel(replace(inp, lfreeb=False), **SMALL)
    x = jnp.asarray(model.x0)
    N, a = map(np.asarray, model._normal_operator(x))
    assert 0.05 < a < 0.5 and N.shape == (48 * 48, x.size)
    rng = np.random.default_rng(0)
    dx = rng.normal(size=x.size) * np.asarray(model.x_scale) * 1e-3
    from vmex.core.optimize import boundary_arrays_from_x

    def points(z):
        rbc, zbs = map(np.asarray, boundary_arrays_from_x(model.fixed, z, model.max_mode, vary_major_radius=True))
        th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 48, endpoint=False),
                             np.linspace(0, 2 * np.pi / int(inp.nfp), 48, endpoint=False), indexing="ij")
        R = sum(rbc[i, m] * np.cos(m * th - (i - (rbc.shape[0] - 1) // 2) * int(inp.nfp) * ph)
                for i in range(rbc.shape[0]) for m in range(rbc.shape[1]))
        Z = sum(zbs[i, m] * np.sin(m * th - (i - (rbc.shape[0] - 1) // 2) * int(inp.nfp) * ph)
                for i in range(rbc.shape[0]) for m in range(rbc.shape[1]))
        return np.stack([R * np.cos(ph), R * np.sin(ph), Z]), th, ph

    X, th, ph = points(np.asarray(x))
    e_t = (np.roll(X, -1, axis=1) - np.roll(X, 1, axis=1)) / (2 * (th[1, 0] - th[0, 0]))
    e_p = (X[:, :, 2:] - X[:, :, :-2]) / (2 * (ph[0, 1] - ph[0, 0]))  # one field period: interior points only
    X, e_t = X[:, :, 1:-1], e_t[:, :, 1:-1]
    n = np.cross(e_t, e_p, axis=0)
    n /= np.linalg.norm(n, axis=0)
    fd = np.sum((points(np.asarray(x) + dx)[0] - points(np.asarray(x) - dx)[0])[:, :, 1:-1] / 2 * n, axis=0)
    np.testing.assert_allclose((N @ dx).reshape(48, 48)[:, 1:-1], fd, atol=2e-2 * np.max(np.abs(fd)))


@needs_vc
@pytest.mark.full
def test_certificate_records_are_complete_and_match_the_row_layout():
    """The recorder sees every certificate round and the outcome: the full Jacobian, rows, coordinates, interior
    state, scales, boundary arrays and the effective resolution; the runtime row layout covers the rows exactly."""
    problem = _small_problem()
    model, records = problem.model, []
    model.recorder = records.append
    sol = problem._solve(problem.x0 + 0.01)
    assert sol is not None
    assert [r["kind"] for r in records][-1] == "certified"
    assert all(r["kind"] == "round" for r in records[:-1]) and len(records) >= 2
    last_row = model.row_blocks[-1][2]
    names = [name for name, _, _ in model.row_blocks]
    assert names[:6] == ["B.n", "pressure", "K_x", "K_y", "K_z", "label"] and "net_current" in names
    for r in records:
        assert r["rows"].size == last_row
        assert r["J"].shape[0] == last_row and r["J"].shape[1] >= model.x0.size
        assert r["x"].size == model.x0.size and r["x_scale"].size == model.x0.size
        assert len(r["state"]) == len(jax.tree.leaves(model.seed[0]))
        assert r["resolution"]["mpol"] == int(model.fixed.mpol) and r["resolution"]["nphi"] == SMALL["nphi"]
        assert r["certificate"] is not None and np.isfinite(r["certificate"]["stationarity"])
    assert "values" in records[0] and records[0]["values"] is not None
    np.testing.assert_array_equal(records[-1]["rows"], sol["rows"])
    model.recorder = None
    problem.close()


def _rotating_current(weight=30.0):
    """Rows (b - 1, b - 1/2, w (c - b^2), w b (c - 1)): the current column (1, b) rotates as b moves."""
    def evaluate(x):
        b, c = x
        return np.array([b - 1.0, b - 0.5, weight * (c - b * b), weight * b * (c - 1.0)]), None

    def jacobian(x):
        b, c = x
        return np.array([[1.0, 0.0], [1.0, 0.0], [-2 * weight * b, weight], [weight * (c - 1.0), weight * b]])

    def conditional_gain(x):
        r = evaluate(x)[0]
        Q, _ = np.linalg.qr(jacobian(x)[:, [1]])
        g = Q.T @ r
        return float(g @ g / (r @ r))

    return evaluate, jacobian, conditional_gain


def test_projection_is_certified_with_fresh_current_columns():
    """Corrections with a stale Jacobian's current columns stop short of conditional stationarity when the columns
    rotate; the finish's fresh-column check refreshes them and certifies the result."""
    evaluate, jacobian, conditional_gain = _rotating_current()
    x0 = np.array([2.0, 4.0])
    r0 = evaluate(x0)[0]
    stale = fvc._levenberg_marquardt(evaluate, x0, r0, None, jacobian(x0), ftol=1e-10, max_nfev=60, verbose=0,
                                     project=np.array([1]))
    assert stale["projection_certified"] is None and conditional_gain(stale["x"]) > 1e-6
    fresh = fvc._levenberg_marquardt(evaluate, x0, r0, None, jacobian(x0), ftol=1e-10, max_nfev=60, verbose=0,
                                     project=np.array([1]), current_columns=lambda x, aux: (jacobian(x)[:, [1]], aux))
    assert fresh["projection_certified"] is True and conditional_gain(fresh["x"]) <= 1e-11


def test_projection_unconverged_is_not_converged():
    """When the fresh-column corrections cannot reach conditional stationarity the finish says so."""
    evaluate, jacobian, _ = _rotating_current()
    x0 = np.array([2.0, 4.0])
    r0 = evaluate(x0)[0]
    out = fvc._levenberg_marquardt(evaluate, x0, r0, None, jacobian(x0), ftol=1e-10, max_nfev=60, verbose=0,
                                   project=np.array([1]),
                                   current_columns=lambda x, aux: (jacobian(x)[:, [1]] + np.array([[0.0], [5.0], [0.0],
                                                                                                   [0.0]]), aux))
    assert out["projection_certified"] is False
    assert out["reason"] == "projection_unconverged" and not out["converged"]


def test_rank_deficient_current_block_rejects():
    evaluate, jacobian, _ = _rotating_current()
    x0 = np.array([2.0, 4.0])
    J = jacobian(x0)
    J[:, 1] = 0.0
    with pytest.raises(RuntimeError, match="rank-deficient current block"):
        fvc._levenberg_marquardt(evaluate, x0, evaluate(x0)[0], None, J, ftol=1e-10, max_nfev=10, verbose=0,
                                 project=np.array([1]))


@pytest.mark.parametrize("rule, converged", [("gain", False), ("model", True)])
def test_stall_rule_model_continues_well_predicted_small_gains(monkeypatch, rule, converged):
    """A linear problem with a large irreducible residual: every step is predicted exactly (gain ratio 1) but gains
    far under 5% of the cost.  The gain rule calls it stalled; the model rule lets the radius grow and converges."""
    def evaluate(x):
        return np.array([x[0] - 1.0, x[1] - 2.0, 100.0]), None

    J = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    x0 = np.array([3.0, -1.0])
    monkeypatch.setattr(fvc, "STALL_RULE", rule)
    out = fvc._levenberg_marquardt(evaluate, x0, evaluate(x0)[0], None, J, ftol=1e-12, max_nfev=60, verbose=0,
                                   radius=1e-3)
    assert out["converged"] is converged
    if converged:
        np.testing.assert_allclose(out["x"], [1.0, 2.0], atol=1e-5)
    else:
        assert out["reason"] == "stalled"


def test_broyden_update_cuts_the_evaluations_of_a_stale_jacobian(monkeypatch):
    """From a stale Jacobian, secant (Broyden) updates along the accepted steps reach the root in far fewer
    evaluations than the fixed Jacobian."""
    def evaluate(x):
        return np.array([x[0] ** 2 - 2.0, x[1] ** 3 - 3.0, x[0] * x[1] - np.sqrt(2) * 3 ** (1 / 3)]), None

    x0 = np.array([3.0, 3.0])
    J0 = np.array([[2 * x0[0], 0.0], [0.0, 3 * x0[1] ** 2], [x0[1], x0[0]]])
    runs = {}
    for mode in ("none", "broyden"):
        monkeypatch.setattr(fvc, "JACOBIAN_UPDATE", mode)
        runs[mode] = fvc._levenberg_marquardt(evaluate, x0, evaluate(x0)[0], None, J0, ftol=1e-14, max_nfev=200,
                                              verbose=0)
        np.testing.assert_allclose(runs[mode]["x"], [np.sqrt(2), 3 ** (1 / 3)], rtol=1e-10)
    assert runs["broyden"]["jacobian_updates"] > 0 and runs["none"]["jacobian_updates"] == 0
    assert runs["broyden"]["nfev"] * 3 < runs["none"]["nfev"]


def test_gauss_newton_extrapolation_follows_a_step_the_model_undershoots(monkeypatch):
    """r(x) = 1/x^2 from x = 1: the Gauss-Newton step (gain ratio 0.80) undershoots a cost that keeps falling along
    it, so doubling the accepted step lowers the cost further for the same evaluations; without the extrapolation
    the fixed Jacobian's steps do not."""
    def evaluate(x):
        return np.array([1.0 / x[0] ** 2]), None

    x0 = np.array([1.0])
    J = np.array([[-2.0]])
    out = {}
    for on in (False, True):
        monkeypatch.setattr(fvc, "GN_EXTRAPOLATE", on)
        out[on] = fvc._levenberg_marquardt(evaluate, x0, evaluate(x0)[0], None, J, ftol=1e-14, max_nfev=6,
                                           verbose=0)
    cost = {on: 0.5 * float(out[on]["rows"] @ out[on]["rows"]) for on in out}
    assert out[False]["extrapolated"] == 0 and out[True]["extrapolated"] >= 2
    assert out[True]["nfev"] <= 6
    assert cost[True] < 0.1 * cost[False]


def test_best_start_ranks_successful_fits_before_cost():
    """start='best': a lower-cost fit that failed never displaces a successful one; among equals, lower cost."""
    from types import SimpleNamespace

    from vmex.core.multigrid import _best_three_term_fit

    good = SimpleNamespace(success=True, cost=2.0)
    failed = SimpleNamespace(success=False, cost=1.0)
    better = SimpleNamespace(success=True, cost=1.5)
    assert _best_three_term_fit([failed, good]) is good
    assert _best_three_term_fit([good, failed, better]) is better
    assert _best_three_term_fit([failed]) is failed
    with pytest.raises(ValueError):
        _best_three_term_fit([])


def test_geodesic_acceleration_follows_curved_rows(monkeypatch):
    """Rosenbrock rows from (0.5, 0) with one fixed Jacobian: the geodesic correction (one extra evaluation for the
    rows' second directional derivative along the step) reaches the root in fewer evaluations than the plain
    Levenberg-Marquardt steps, which stall short of it."""
    def evaluate(x):
        return np.array([x[0] - 1.0, 10.0 * (x[1] - x[0] ** 2)]), None

    x0 = np.array([0.5, 0.0])
    J = np.array([[1.0, 0.0], [-20.0 * x0[0], 10.0]])
    out = {}
    for on in (False, True):
        monkeypatch.setattr(fvc, "GEODESIC", on)
        out[on] = fvc._levenberg_marquardt(evaluate, x0, evaluate(x0)[0], None, J, ftol=1e-14, max_nfev=30, verbose=0)
    assert out[False]["accelerated"] == 0 and out[True]["accelerated"] >= 1
    assert out[True]["converged"] and np.allclose(out[True]["x"], [1.0, 1.0], atol=1e-6)
    assert out[True]["nfev"] < out[False]["nfev"]
