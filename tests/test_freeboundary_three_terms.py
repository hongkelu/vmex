"""Free boundary from the three virtual-casing interface conditions.

:mod:`vmex.core.freeboundary_three_terms` solves for the boundary on which ``B.n``,
the pressure jump and the sheet current vanish. Lanes: input validation and the
configuration's deck keys; the residual rows' definitions on a converged finite-beta state; and (``full``)
the vacuum and finite-beta Landreman--Paul QA cases with their bundled ESSOS
coils, where the solve agrees with NESTOR and is independent of its start.
"""

from __future__ import annotations

from dataclasses import replace
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_three_terms as fvc  # noqa: E402
from vmex.core import freeboundary_three_terms_problem as ftp  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402
from vmex.core.errors import TrialRejected  # noqa: E402
from vmex.core.freeboundary_three_terms_problem import ThreeTermsFreeBoundaryProblem  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
needs_vc = pytest.mark.skipif(not vc._HAVE_VCJ, reason="requires virtual_casing_jax")


def _deck(name, mpol, ns):
    inp = vj.VmecInput.from_file(DATA / name).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=16)
    return replace(inp, lfreeb=True, mgrid_file="essos_coils(direct)", ns_array=np.array([ns]),
                   niter_array=np.array([5000]), ftol_array=np.array([1e-10]), delt=0.5)


def _coil_field(name):
    """The bundled ESSOS coils' exact filament field (``essos.fields.BiotSavart``, a pytree: no mgrid tabulation)."""
    pytest.importorskip("essos")
    from essos.coils import Coils
    from essos.fields import BiotSavart

    return BiotSavart(Coils.from_json(str(DATA / name)))


def _scaled(field, factor):
    """``field`` with every coil current multiplied by ``factor``."""
    from essos.coils import Coils
    from essos.fields import BiotSavart

    return BiotSavart(Coils(field.coils.curves, field.coils.currents * factor))


def test_nonzero_edge_pressure_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), pmass_type="power_series",
                  am=np.array([1.0e3, -0.5e3] + [0.0] * 19))
    with pytest.raises(ValueError, match="edge pressure"):
        fvc.solve_free_boundary_three_terms(inp, external_field=lambda xyz: xyz)


def test_asymmetric_boundary_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), lasym=True)
    with pytest.raises(NotImplementedError, match="lasym"):
        fvc.solve_free_boundary_three_terms(inp, external_field=lambda xyz: xyz)


def test_three_terms_config_validates_and_follows_the_deck(tmp_path):
    """The configuration's choices and default weights, ``"none"`` as the anchored labelling without rows, and the
    deck keys: ``None`` follows ``LTHREE_TERMS``, ``True`` is the defaults, ``False`` disables; the keys round-trip
    through an INDATA file."""
    for bad, match in ((dict(labelling="condensed"), "labelling"), (dict(start="coils"), "start"),
                       (dict(labelling="none", label_weight=1e-2), "none")):
        with pytest.raises(ValueError, match=match):
            vj.ThreeTermsConfig(**bad)
    weights = {name: vj.ThreeTermsConfig(labelling=name).resolved_label_weight
               for name in ("anchored", "equal_arc", "none")}
    assert weights == {"anchored": 1e-2, "equal_arc": 1e-3, "none": 0.0}
    options = vj.ThreeTermsConfig(labelling="none").model_options()
    assert options["labelling"] == "none" and options["label_weight"] == 0.0
    # at weight 0 every labelling is the no-label model
    assert all(vj.ThreeTermsConfig(labelling=name, label_weight=0.0).model_options() == options
               for name in ("anchored", "equal_arc"))
    inp = _deck("input.LandremanPaul2021_QA_lowres", 3, 11)
    assert fvc.resolve_three_terms(None, inp) is None and fvc.resolve_three_terms(True, inp) == vj.ThreeTermsConfig()
    deck = replace(inp, lthree_terms=True, three_terms_labelling="ANCHORED", three_terms_label_weight=-1.0)
    config = fvc.resolve_three_terms(None, deck)
    assert config.labelling == "anchored" and config.label_weight is None
    assert fvc.resolve_three_terms(False, deck) is None
    read = vj.VmecInput.from_file(replace(deck, three_terms_label_weight=0.05).to_indata(tmp_path / "input.three"))
    assert (read.lthree_terms, read.three_terms_labelling, read.three_terms_label_weight) == (True, "anchored", 0.05)
    assert not vj.VmecInput.from_file(inp.to_indata(tmp_path / "input.nestor")).lthree_terms
    with pytest.raises(ValueError, match="THREE_TERMS_LABELLING"):
        replace(inp, three_terms_labelling="gauge")


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
#: ``SMALL`` with the anchored labelling at its default weight (the labelling these tests were written for).
SMALL_ANCHORED = vj.ThreeTermsConfig(labelling="anchored", **SMALL)


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
    result = vj.solve_free_boundary_multigrid(inp, external_field=field,
                                              three_terms=replace(SMALL_ANCHORED, max_nfev=20))
    res = result.three_terms.boundary_residual
    assert result.vacuum is None and res.normal < design.normal and res.sheet_current < design.sheet_current
    assert result.three_terms.start in ("deck", "nestor") and result.three_terms.config.labelling == "anchored"


@needs_vc
@pytest.mark.full
def test_uncertified_tangent_solve_is_no_jacobian(monkeypatch):
    """A boundary Jacobian whose tangent solves are not all certified raises instead of being used."""
    from vmex.core import implicit as im

    inp, field = _small_vacuum()
    model = fvc.ThreeTermsModel(replace(inp, lfreeb=False), **SMALL)
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
    inp, field = _small_vacuum()
    problem = ThreeTermsFreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(1),
        field_from_parameters=lambda x: _scaled(field, 1 + 0.05 * x[0]),
        quantities=(opt.mean_iota,), config=SMALL_ANCHORED)
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
    model = fvc.ThreeTermsModel(replace(inp, lfreeb=False), **SMALL)
    monkeypatch.setattr(model, "evaluate", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="no certified equilibrium"):
        model.solve_boundary(model.params0, field)


@needs_vc
@pytest.mark.full
@pytest.mark.parametrize("label_weight", [1e-3, 1e-2])
def test_equal_arc_problem_gradient_matches_resolved_differences(label_weight):
    """The equal-arc labelling's loss gradient and constraint Jacobian on the small vacuum design, against
    re-solved central differences at two step sizes, with tight fits."""
    problem = _small_problem(labelling="equal_arc", label_weight=label_weight, boundary_ftol=1e-10,
                             boundary_max_nfev=80)
    x0 = problem.x0
    value, grad = problem.value_and_grad(x0)
    jac = problem.constraint_jac(x0)
    for h in (1e-3, 3e-4):
        fd = [(problem.fun(x0 + h) - problem.fun(x0 - h)) / (2 * h),
              (problem.constraint_values(x0 + h) - problem.constraint_values(x0 - h)) / (2 * h)]
        np.testing.assert_allclose(grad, fd[0], rtol=1e-2)
        np.testing.assert_allclose(jac[:, 0], fd[1], rtol=1e-2)
    problem.close()


def _small_problem(**options):
    """A one-coil-current design on the small vacuum free boundary: aspect ratio, with mean iota as a quantity.
    ``boundary_*`` options are the problem's, the others the configuration's (``SMALL_ANCHORED``)."""
    inp, field = _small_vacuum()
    fit = {name: options.pop(name) for name in list(options) if name.startswith("boundary_")}
    return ThreeTermsFreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(1),
        field_from_parameters=lambda x: _scaled(field, 1 + 0.05 * x[0]),
        quantities=(opt.mean_iota,), config=replace(SMALL_ANCHORED, **options), **fit)


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


def test_finalize_re_solves_the_current_on_the_tightened_state():
    """A fit closes the stiff (bootstrap) row on its trial-tolerance state; tightened, that row reopens.  The trial's
    record re-solves the current coordinate there, so the certificate sees the boundary's stationarity, not the
    current's loose-state offset (QH 2.5%: 92% of the tightened cost)."""
    class Model:
        n_boundary, x0, x_scale, _projected = 1, np.zeros(2), np.ones(2), np.array([1])

        @staticmethod
        def rows(x, tight):  # interface, an irreducible row, the current's closure (offset at the trial tolerance)
            return np.array([x[0] - 1.0, 1e-3, 100.0 * (x[1] - x[0] ** 2) + (0.0 if tight else 1e-3)])

        def evaluate(self, x, params, field, seed=None, tight=False):
            return self.rows(x, tight), (np.array(x, dtype=float), None, None, tight)

        def linearize(self, x, aux, field, extra=None):
            J = np.array([[1.0, 0.0], [0.0, 0.0], [-200.0 * x[0], 100.0]])
            return J, None, None, (np.array(x, dtype=float), None, None, True)

        def _rows(self, state, params, field):
            return self.rows(state, True)

        def solve_boundary(self, params, field, *, x0, **kwargs):  # a polish that gains nothing
            return dict(x=x0, rows=self.rows(x0, True), aux=(np.array(x0), None, None, True), nfev=1)

    problem = object.__new__(ThreeTermsFreeBoundaryProblem)
    problem.model, problem._plasma = Model(), np.zeros(0)
    problem._rows = lambda state, params, x: np.zeros(1)
    problem.stats = dict(linearizations=0, max_condition=0.0, boundary_evaluations=0)
    problem.boundary_ftol, problem.boundary_xtol, problem.boundary_max_residual = 1e-4, None, None
    problem._checkpoint = lambda: None
    x = np.array([1.0, 1.0 - 1e-5])  # the fit's point: the closure row is zero on the trial state only
    fit = dict(x=x, rows=Model.rows(x, False), aux=(x, None, None, False), nfev=1, reason="ftol", converged=True)
    sol = problem._finalize(np.zeros(1), None, None, fit)
    assert sol["certificate"]["stationarity"] < 1e-8
    np.testing.assert_allclose(sol["x"], [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(sol["rows"], [0.0, 1e-3, 0.0], atol=1e-12)


def test_fit_certificate_measures_stationarity_and_rank():
    """The certificate's stationarity is the Gauss-Newton step's predicted relative gain; a repeated column is a
    rank-deficient Jacobian."""
    J = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    assert ftp._fit_certificate(J, np.array([0.0, 0.0, 2.0]))["stationarity"] == pytest.approx(0.0)
    assert ftp._fit_certificate(J, np.array([1.0, 0.0, 1.0]))["stationarity"] == pytest.approx(0.5)
    flat = ftp._fit_certificate(np.diag([1.0, 1e-3, 0.0])[:, :2], np.array([1e-6, 1e-6, 1.0]))
    assert flat["stationarity"] < 1e-11 and flat["step"] == pytest.approx(1e-3)  # flat: tiny gain, long step
    certificate = ftp._fit_certificate(np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]), np.ones(3))
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
    inp, field = _small_vacuum()
    with pytest.raises(ValueError, match="edge pressure"):
        fvc.ThreeTermsModel(replace(inp, lfreeb=False, am=np.r_[1.0e3, np.zeros(20)]), **SMALL)
    problem = ThreeTermsFreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: field,
        plasma_from_parameters=lambda params, x: replace(params, am=params.am.at[0].add(1.0e3 * x[1])),
        config=SMALL_ANCHORED)
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
    offset, not with the updated PHIEDGE as its base, and through x2 although its derivative is zero at x0.
    Tight fits: the gradient is the implicit derivative of a stationary fit, and the default fits' trials stop
    short of it (their differences are 1.8% off; at ftol 1e-6 they agree to 5e-4)."""
    inp, field = _small_vacuum()
    problem = ThreeTermsFreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(3),
        field_from_parameters=lambda x: _scaled(field, 1 + 0.05 * x[0]),
        plasma_from_parameters=lambda params, x: replace(
            params, phiedge=params.phiedge * (1 + 0.5 * x[1]) + 0.01 * x[2] ** 2),
        config=SMALL_ANCHORED, boundary_ftol=1e-6, boundary_max_nfev=200)
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
    perturbed seed gives the same free-boundary physics: interface residuals and boundary Jacobian agree far
    inside the truncation level, through the interior's frozen m = 1 complement and the hot restart.  The
    pointwise rows are not compared exactly: the interior angle along the frozen m = 1 complement is nearly
    flat, so the converged states differ along it (rows by ~0.2% here, B.n RMS by 1e-4 relative, up to 0.4%
    at mpol 3 and 0.3% at mpol 4, where truncation changes B.n twofold from 8 to 12 modes)."""
    inp, field = _small_vacuum()
    model = fvc.ThreeTermsModel(replace(inp, lfreeb=False), **SMALL)
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
        results.append((np.asarray(rows), np.asarray(J), model.boundary_residual(aux[0], aux[2], field)))
        model._last_linearize = None
    rows0, J0, residual0 = results[0]
    for rows, J, residual in results[1:]:
        for name in ("normal", "pressure", "sheet_current"):
            assert getattr(residual, name) == pytest.approx(getattr(residual0, name), rel=1e-3), name
        assert np.linalg.norm(J - J0) <= 1e-3 * np.linalg.norm(J0)
        assert np.linalg.norm(rows - rows0) <= 1e-2 * np.linalg.norm(rows0)


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
    with pytest.raises(ValueError, match="model"):
        _small_problem(label_weight=0.0).restore_checkpoint(path)
    other, field = _small_problem(), _small_vacuum()[1]  # same settings, another coil-field map
    other.field = lambda x: _scaled(field, 1 + 0.06 * x[0])
    with pytest.raises(ValueError, match="coil field"):
        other.restore_checkpoint(path)
    # and in a fresh process: nothing but the checkpoint carries the state over.  It shares this process's
    # device, so it must not preallocate (a GPU this process preallocated has no room for a second pool).
    import os
    import subprocess
    import sys

    script = ("import sys; sys.path.insert(0, 'tests'); from test_freeboundary_three_terms import _small_problem; "
              f"print('change', _small_problem().restore_checkpoint({str(path)!r}))")
    done = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1], capture_output=True,
                          text=True, check=True, env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"))
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
    free = vj.solve_free_boundary_multigrid(inp, external_field=field)
    wout = vj.wout_from_state(inp=inp, state=free.state, fsqr=float(free.fsqr), fsqz=float(free.fsqz),
                              fsql=float(free.fsql), niter=int(free.iterations), converged=bool(free.converged),
                              vacuum_output=free.vacuum)
    rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
    for m, n, r, z in zip(np.asarray(wout.xm, int), np.asarray(wout.xn, int) // int(wout.nfp),
                          np.asarray(wout.rmnc)[-1], np.asarray(wout.zmns)[-1]):
        rbc[n + inp.ntor, m], zbs[n + inp.ntor, m] = r, z
    deck = replace(inp, rbc=rbc, zbs=zbs, lfreeb=False, raxis_c=np.asarray(wout.raxis_cc)[:inp.ntor + 1],
                   zaxis_s=np.asarray(wout.zaxis_cs)[:inp.ntor + 1])
    return opt.solve_equilibrium(deck, initial_state=free.state), deck


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
    nestor, nestor_deck = _nestor(inp, field)
    before = _residual(inp, nestor, field)
    # The net-current row's reference: outside the plasma the toroidal circulation is the coils' alone, so a
    # free boundary's edge R B_phi equals the coil current that _coil_net_current integrates on the axis.
    np.testing.assert_allclose(abs(fvc._coil_net_current(field, nestor.wout)), abs(float(nestor.wout.rbtor)),
                               rtol=1e-3)
    config = vj.ThreeTermsConfig(labelling="anchored")
    fits = [fvc.solve_free_boundary_three_terms(inp, config, external_field=field, initial_boundary=start)
            for start in (None, nestor_deck)]
    for fit in fits:
        after = _residual(inp, fit.equilibrium, field)
        assert after.normal < before.normal and after.sheet_current < before.sheet_current
        assert fit.converged and fit.success and "interface residuals" in fit.message
    strict = fvc.solve_free_boundary_three_terms(inp, replace(config, max_residual=1e-6), external_field=field,
                                                 previous=fits[0])
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
        vary_major_radius=True, jacobian_batch_size=8, device=jax.devices()[0], restart_from=model.seed[0])
    J = model.linearize(model.x0, (*model.seed, model.params0, True), field)[0]
    J_upstream = np.asarray(problem.residual_jac(model.x0))
    assert np.linalg.norm(J - J_upstream) < 1e-9 * np.linalg.norm(J_upstream)
    again = fvc.solve_free_boundary_three_terms(inp, config, external_field=field, previous=fit)
    # The restart accepts the fit without a Jacobian; its hot-restarted equilibrium re-solve moves the cost at
    # that solve's tolerance (0.3% on GPU), not the fit's.
    assert again.njev == 0 and again.cost <= fit.cost * (1 + 1e-2)
    assert _lcfs_distance(again.equilibrium.wout, walls[0]) < 1e-4


@needs_vc
@pytest.mark.full
def test_finite_beta_free_boundary_removes_the_sheet_current():
    """LP QA at 0.5% beta with its fitted coils, through the solver's public option."""
    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    design = _residual(inp, opt.solve_equilibrium(replace(inp, lfreeb=False)), field)
    result = vj.solve_free_boundary_multigrid(inp, external_field=field,
                                              three_terms=vj.ThreeTermsConfig(labelling="anchored"))
    assert result.converged and result.vacuum is None
    res = result.three_terms.boundary_residual
    for name in ("normal", "pressure", "sheet_current"):
        assert getattr(res, name) < 0.25 * getattr(design, name), name
    assert res.sheet_current < 1.5e-3


@needs_vc
@pytest.mark.full
@pytest.mark.parametrize("labelling, label_weight", [("anchored", None), ("equal_arc", 1e-3), ("equal_arc", 1e-2)])
def test_three_term_problem_gradients_match_resolved_differences(labelling, label_weight):
    """Design gradients by the implicit function theorem of the boundary fit, against re-solved central
    differences, along a plasma coordinate (PHIEDGE) and a field coordinate (the coil currents), for the start
    labelling and for equal arc at its default weight and at 1e-2 (tight fits: at a small weight the labelling is
    pinned only to the fit's tolerance)."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 13), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    problem = ThreeTermsFreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: _scaled(coils, 1 + x[1]),
        plasma_from_parameters=lambda params, x: replace(params, phiedge=params.phiedge * (1 + x[0])),
        quantities=(opt.mean_iota,),
        parameter_quantities=(lambda state, runtime, x: opt.aspect_ratio(state, runtime) * (1 + x[1]),),
        config=vj.ThreeTermsConfig(labelling=labelling, label_weight=label_weight, nphi=24, ntheta=24,
                                   quadrature=(4 * 2 * 24, 48)),
        boundary_ftol=1e-10, boundary_max_nfev=80)
    assert isinstance(problem, ThreeTermsFreeBoundaryProblem)
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
    fit = fvc.solve_free_boundary_three_terms(
        inp, vj.ThreeTermsConfig(labelling="anchored", bootstrap=profiles, jacobian_ftol=None), external_field=field)
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
    model = fvc.ThreeTermsModel(inp, bootstrap=profiles, fixed_boundary=True)
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
    three = vj.solve_free_boundary_multigrid(inp, three_terms=vj.ThreeTermsConfig(labelling="anchored"))
    assert three.converged
    w = _wout(inp, three)
    vmec2000 = SimpleNamespace(nfp=int(w.nfp), xm=np.asarray(reference["xm"]), xn=np.asarray(reference["xn"]),
                               rmnc=np.asarray([reference["rmnc"]]), zmns=np.asarray([reference["zmns"]]))
    print(f"{case}: three-term LCFS {1e3 * _lcfs_distance(w, vmec2000):.2f} mm from VMEC2000's; residuals "
          f"{three.three_terms.boundary_residual} against VMEC2000's {reference['residual']}")
    for name in ("normal", "sheet_current"):
        assert getattr(three.three_terms.boundary_residual, name) < reference["residual"][name]
    if case == "vacuum":
        field = MgridField.from_input(inp, CTH_MGRID, order=3)  # the smooth field the three-term solve uses
        (bn, drift), (bn_ref, drift_ref) = _vacuum_flux_surface(w, field), _vacuum_flux_surface(vmec2000, field)
        print(f"vacuum: coil B.n {bn:.2e} (VMEC2000 {bn_ref:.2e}), field-line drift {1e3 * drift:.2f} mm "
              f"(VMEC2000 {1e3 * drift_ref:.2f} mm)")
        assert bn < 5e-4 and bn < bn_ref
        assert drift < 1e-3  # absolute: at 5 x 4 modes VMEC2000's boundary drifts a little less (see above)
