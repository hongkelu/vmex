"""Single-stage optimization on the three-term free boundary.

:class:`vmex.core.freeboundary_three_terms_problem.ThreeTermsFreeBoundaryProblem`:
the definitions it refuses, its seed perturbations, and (``full``) a
coil-current design on the small vacuum Landreman--Paul QA free boundary:
exact gradients against re-solved differences and one SLSQP step, one record
per design, rejected trials that leave the continuation alone, and checkpoints.
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_three_terms as fbt  # noqa: E402
from vmex.core import freeboundary_three_terms_implicit as fbti  # noqa: E402
from vmex.core.errors import TrialRejected  # noqa: E402
from vmex.core.freeboundary_three_terms_problem import ThreeTermsFreeBoundaryProblem  # noqa: E402

from tests.test_freeboundary_three_terms import (  # noqa: E402
    SMALL, _coil_field, _deck, _lp_beta0p5_profiles, _small_vacuum, needs_vc)

_CONFIG_FIELDS = {f.name for f in fields(vj.ThreeTermsConfig)}


def _small_problem(**options):
    """A one-coil-current design on the small vacuum free boundary: aspect ratio, with mean iota as a quantity.
    ``options`` are ThreeTermsConfig fields, from_loss keywords or model keywords."""
    inp, field = _small_vacuum()
    base = field.currents
    config = vj.ThreeTermsConfig(**{**SMALL, **{k: v for k, v in options.items() if k in _CONFIG_FIELDS}})
    return ThreeTermsFreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(1),
        field_from_parameters=lambda x: replace(field, currents=base * (1 + 0.05 * x[0])),
        quantities=(opt.mean_iota,), config=config,
        **{k: v for k, v in options.items() if k not in _CONFIG_FIELDS})


def test_problem_needs_the_gauge_and_callables():
    """The single-stage problem differentiates the converged point on the gauge: "none" labelling, non-callable
    maps and quantities are refused before any solve."""
    inp = _deck("input.LandremanPaul2021_QA_lowres", 3, 11)
    with pytest.raises(ValueError, match="labelling='gauge'"):
        ThreeTermsFreeBoundaryProblem.from_loss(inp, lambda s, r, x: 0.0, np.zeros(1),
                                               field_from_parameters=lambda x: None,
                                               config=vj.ThreeTermsConfig(labelling="none"))
    with pytest.raises(TypeError, match="callable"):
        ThreeTermsFreeBoundaryProblem.from_loss(inp, lambda s, r, x: 0.0, np.zeros(1), field_from_parameters=None)
    assert vj.ThreeTermsFreeBoundaryProblem is ThreeTermsFreeBoundaryProblem


def test_three_terms_problem_gradient_and_one_slsqp_step():
    """A coil-current design at low resolution: the loss gradient and constraint Jacobian against re-solved central
    differences, one SLSQP step through ``minimize``, and the accepted point's reports."""
    problem = _small_problem()
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
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # the report re-solves the accepted state, not another one
        assert problem.equilibrium_from_x(x).wout.aspect == pytest.approx(result.fun, rel=1e-6)
    problem.close()
    with pytest.raises(ValueError, match="evaluate the candidate"):
        problem.accept_x(x0 + 0.123)
    assert problem.stats["trials"] >= 3 and problem.stats["linearizations"] >= 1


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


@needs_vc
@pytest.mark.full
def test_unconverged_fits_and_rank_deficient_jacobians_reject_the_trial(monkeypatch):
    """A boundary fit that did not converge is no start of a trial; a rank-deficient boundary Jacobian is no
    record; a certified gradient comes with the value of the same record."""
    problem = _small_problem()
    model = problem.model
    value, gradient = problem.value_and_grad(problem.x0)
    assert problem.fun(problem.x0) == value
    solve_boundary = model.solve_boundary
    monkeypatch.setattr(model, "solve_boundary",
                        lambda *args, **kwargs: dict(solve_boundary(*args, **kwargs), converged=False, reason="radius"))
    assert problem._solve(problem.x0 + 0.01, starts=[None]) is None  # a fitted start only: no direct Newton
    assert problem.stats["fit_reasons"]["radius"] >= 1
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
    """With looser trial equilibria (trial_ftol), a trial's value is that of the state its gradient uses (values from loose
    states would make derivatives look wrong against them)."""
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
    inp, field = _small_vacuum()
    with pytest.raises(ValueError, match="edge pressure"):
        fbt.ThreeTermsFreeBoundaryModel(replace(inp, lfreeb=False, am=np.r_[1.0e3, np.zeros(20)]), **SMALL)
    problem = ThreeTermsFreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: field,
        plasma_from_parameters=lambda params, x: replace(params, am=params.am.at[0].add(1.0e3 * x[1])),
        config=vj.ThreeTermsConfig(**SMALL))
    assert problem._solve(np.array([0.0, 0.01])) is None
    with pytest.raises(TrialRejected, match="edge pressure"):
        problem.fun(np.array([0.0, 0.01]))
    problem.close()


def _short_trial_fits(monkeypatch, model):
    """Trial boundary fits that end short of their stationary point (the Newton then moves them), as the
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
    """boundary_max_residual judges the state the values come from (Newton-converged and re-solved tight), not the
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
    assert problem._solve(problem.x0 + 0.01, starts=[None]) is None
    assert problem._solve(problem.x0 + 0.02, starts=[None]) is None
    (radius_1, x_1), (radius_2, x_2) = seen[0], seen[-1]
    assert radius_2 == radius_1 != 123.0
    assert (x_1 is None and x_2 is None) or np.array_equal(x_1, x_2)
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
    assert resumed.refine_accepted() <= 1e-6  # the restored record is already Newton-converged
    with pytest.raises(ValueError, match="model"):
        _small_problem(label_weight=1e-2).restore_checkpoint(path)
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

    script = ("import sys; sys.path.insert(0, '.'); from tests.test_freeboundary_three_terms_problem import "
              f"_small_problem; print('change', _small_problem().restore_checkpoint({str(path)!r}))")
    done = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1], capture_output=True,
                          text=True, check=True)
    assert float(done.stdout.split("change")[-1]) <= 1e-6


@needs_vc
@pytest.mark.full
def test_three_terms_problem_gradients_match_resolved_differences():
    """Exact design gradients against re-solved central differences, along a plasma coordinate (PHIEDGE) and a
    field coordinate (the coil currents); a trial whose certified residual is above boundary_max_residual is no
    solution."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 13), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    problem = ThreeTermsFreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: replace(coils, currents=coils.currents * (1 + x[1])),
        plasma_from_parameters=lambda params, x: replace(params, phiedge=params.phiedge * (1 + x[0])),
        quantities=(opt.major_radius,),
        parameter_quantities=(lambda state, runtime, x: opt.aspect_ratio(state, runtime) * (1 + x[1]),),
        boundary_ftol=1e-10, boundary_max_nfev=80,
        config=vj.ThreeTermsConfig(nphi=24, ntheta=24, quadrature=(4 * 2 * 24, 48)))
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


@needs_vc
@pytest.mark.full
def test_bootstrap_current_on_the_gauge_has_the_exact_design_gradient():
    """LP QA at 0.5% beta with the Redl current solved in the equilibrium, on the exact gauge: the Newton polish
    runs on the gauge with the current projected (the boundary on the gauge to GAUGE_TOL, the current
    self-consistent), and the exact design gradient (its multipliers from the projected Hessian, never the joint
    one) equals central differences of re-solved designs."""
    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    profiles = _lp_beta0p5_profiles(inp)
    base = field.currents
    problem = ThreeTermsFreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(1),
        field_from_parameters=lambda x: replace(field, currents=base * (1 + 0.02 * x[0])),
        config=vj.ThreeTermsConfig(labelling="gauge"), bootstrap=profiles)
    x0 = problem.x0
    value, grad = problem.value_and_grad(x0)
    sol = problem._record(x0)
    assert sol["newton"]["projected"] and sol["newton"]["gauge"] == "condensation" and sol["newton"]["converged"]
    model = problem.model
    assert float(np.abs(np.asarray(model.gauge_rows(np.asarray(sol["x"])))).max()) <= 10 * fbti.GAUGE_TOL
    state, params = problem.state(x0)
    assert model.bootstrap_residual(state, params) < 1e-2
    h = 1e-3
    fd = (problem.fun(x0 + h) - problem.fun(x0 - h)) / (2 * h)
    np.testing.assert_allclose(grad, fd, rtol=2e-3)
    problem.close()


def test_from_tuples_assembles_the_loss_rows_and_bounds(monkeypatch):
    """``from_tuples``: upstream's (function, target, weight) terms make the loss, and each constraint's bounds and
    scale stay with its row (checked without a free-boundary solve: ``from_loss`` is replaced)."""
    captured = {}

    class Built:
        nonlinear_constraint = staticmethod(lambda lower, upper, scales: (list(lower), list(upper), list(scales)))

    def fake_from_loss(cls, inp, loss, x0, *, quantities, parameter_quantities, **kwargs):
        captured.update(loss=loss, quantities=quantities, parameter_quantities=parameter_quantities, kwargs=kwargs)
        return Built()

    monkeypatch.setattr(ThreeTermsFreeBoundaryProblem, "from_loss", classmethod(fake_from_loss))
    aspect = lambda state, runtime: jnp.asarray(state) * 2.0  # noqa: E731
    vector = lambda state, runtime: jnp.asarray([1.0, 3.0])  # noqa: E731
    iota = lambda state, runtime: jnp.asarray(0.4)  # noqa: E731
    distance = lambda state, runtime, x: jnp.asarray(0.3)  # noqa: E731
    problem = ThreeTermsFreeBoundaryProblem.from_tuples(
        None, [(aspect, 6.0, 0.5), (vector, 1.0, 2.0)], np.zeros(2), constraints=[(iota, 0.42, np.inf)],
        parameter_constraints=[(distance, 0.2, np.inf, 0.1)], design_constraints=["coil limits"],
        field_from_parameters=lambda x: None)
    # 0.5 * 0.5 * (2 * 4 - 6)^2 + 0.5 * 2 * ((1 - 1)^2 + (3 - 1)^2) = 1 + 4
    assert float(captured["loss"](4.0, None, None)) == pytest.approx(5.0)
    assert captured["quantities"] == (iota,) and captured["parameter_quantities"] == (distance,)
    assert ThreeTermsFreeBoundaryProblem.constraints(problem) == [([0.42, 0.2], [np.inf, np.inf], [1.0, 0.1]),
                                                             "coil limits"]
    with pytest.raises(TypeError, match="constraints"):
        ThreeTermsFreeBoundaryProblem.from_tuples(None, [(aspect, 6.0, 0.5)], np.zeros(2), constraints=[(iota, 0.4)],
                                                  field_from_parameters=lambda x: None)
    with pytest.raises(ValueError, match="lower <= upper"):
        ThreeTermsFreeBoundaryProblem.from_tuples(None, [(aspect, 6.0, 0.5)], np.zeros(2),
                                                  constraints=[(iota, 0.5, 0.4)], field_from_parameters=lambda x: None)
