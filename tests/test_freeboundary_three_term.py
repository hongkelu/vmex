"""Free boundary from the three virtual-casing interface conditions.

:mod:`vmex.core.freeboundary_three_term` solves for the boundary on which
``B.n``, the pressure jump and the sheet current vanish. Lanes: input
validation, the config and its INDATA keys; the residual rows' definitions on a
converged finite-beta state; the boundary fit's Levenberg-Marquardt steps and
certificate on analytic least squares; and (``full``) the vacuum and
finite-beta Landreman--Paul QA cases with their bundled ESSOS coils, where the
solve agrees with NESTOR and is independent of its start.  The gauge Newton and
the exact derivatives are tests/test_freeboundary_three_term_implicit.py, the
single-stage problem tests/test_freeboundary_three_term_problem.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
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
from vmex.core import freeboundary_three_term as fbt  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402

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


#: The smallest vacuum case that still has a free boundary: LP QA at mpol = ntor = 2, ns 7, a 12 x 12 interface
#: grid with the boundary modes up to 1, and its fixed singular quadrature (4 nfp nphi, 2 ntheta).
SMALL = dict(nphi=12, ntheta=12, max_mode=1, quadrature=(96, 24))


def _small_vacuum():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 2, 7), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    return inp, _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")


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
    return fbt.summarize_boundary_residual(*fbt.boundary_residual(
        inp, eq.solution, field, runtime=eq.solver_context, nphi=n, ntheta=n))


def _wout(inp, result):
    return vj.wout_from_state(inp=inp, state=result.state, fsqr=float(result.fsqr), fsqz=float(result.fsqz),
                              fsql=float(result.fsql), converged=bool(result.converged), vacuum_output=result.vacuum)


def _equilibrium(result):
    """A three-term result's free boundary as an ``Equilibrium`` (the fixed-boundary deck and the returned state)."""
    return opt.solve_equilibrium(result.three_term.input, initial_state=result.state)


def _cost(result, field):
    """``|r|^2 / 2`` of the interface rows at a three-term result's boundary (a tight re-evaluation)."""
    tt = result.three_term
    rows = np.asarray(tt.model.evaluate(tt.x, tt.model.params0, field, tight=True)[0])
    return 0.5 * float(rows @ rows)


def test_nonzero_edge_pressure_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), pmass_type="power_series",
                  am=np.array([1.0e3, -0.5e3] + [0.0] * 19))
    with pytest.raises(ValueError, match="edge pressure"):
        fbt.solve_free_boundary_three_term(inp, external_field=lambda xyz: xyz)


def test_asymmetric_boundary_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), lasym=True)
    with pytest.raises(NotImplementedError, match="lasym"):
        fbt.solve_free_boundary_three_term(inp, external_field=lambda xyz: xyz)


def test_config_validates_and_resolves_like_position_control(tmp_path):
    """ThreeTermConfig checks its choices and fills the Newton defaults; ``three_term=None`` follows the deck's
    LTHREE_TERM / THREE_TERM_LABELLING (written and read back by the INDATA writer), ``False`` is off, ``True`` the
    deck's labelling; the NESTOR time-stepping controls are refused with it."""
    with pytest.raises(ValueError, match="labelling"):
        vj.ThreeTermConfig(labelling="start")
    with pytest.raises(ValueError, match="fit"):
        vj.ThreeTermConfig(fit="lm")
    with pytest.raises(ValueError, match="m1_complement"):
        vj.ThreeTermConfig(m1_complement="solved")
    config = vj.ThreeTermConfig(newton=dict(verbose=True))
    assert config.newton == dict(step_tol=1e-8, max_steps=60, quasi=True, verbose=True)
    assert config.model_options()["labelling"] == "gauge"
    inp = _deck("input.LandremanPaul2021_QA_lowres", 3, 11)
    assert fbt.resolve_three_term(None, inp) is None and fbt.resolve_three_term(False, inp) is None
    assert fbt.resolve_three_term(True, inp) == vj.ThreeTermConfig()
    assert fbt.resolve_three_term(config, inp) is config
    with pytest.raises(TypeError, match="three_term"):
        fbt.resolve_three_term("gauge", inp)
    deck = replace(inp, lthree_term=True, three_term_labelling="none")
    path = deck.to_indata(tmp_path / "input.three_term")
    text = path.read_text()
    assert "LTHREE_TERM" in text and "THREE_TERM_LABELLING" in text
    read = vj.VmecInput.from_file(path)
    assert read.lthree_term and read.three_term_labelling == "none"
    assert fbt.resolve_three_term(None, read) == vj.ThreeTermConfig(labelling="none")
    with pytest.raises(ValueError, match="THREE_TERM_LABELLING"):
        replace(inp, three_term_labelling="start")
    with pytest.raises(ValueError, match="time_step"):
        vj.solve_free_boundary_multigrid(deck, external_field=lambda xyz: xyz, time_step=0.5)


def test_multigrid_keyword_dispatches_to_the_three_term_solve(monkeypatch):
    """``three_term=`` (or the deck) hands the ladder, the field, the start and the config to
    solve_free_boundary_three_term; an unconverged result raises with raise_on_max_iterations."""
    calls = []

    def fake(inp, **kwargs):
        calls.append((inp, kwargs))
        return SimpleNamespace(converged=kwargs["config"].labelling == "none",
                               three_term=SimpleNamespace(message="not certified"))

    monkeypatch.setattr(fbt, "solve_free_boundary_three_term", fake)
    inp = _deck("input.LandremanPaul2021_QA_lowres", 3, 11)
    field = object()
    got = vj.solve_free_boundary_multigrid(inp, ns_array=[5, 9], external_field=field,
                                           three_term=vj.ThreeTermConfig(labelling="none"))
    (ladder, kwargs), = calls
    assert got.converged and kwargs["external_field"] is field and kwargs["config"].labelling == "none"
    np.testing.assert_array_equal(ladder.ns_array, [5, 9])
    with pytest.raises(vj.VmecConvergenceError, match="not certified"):
        vj.solve_free_boundary_multigrid(replace(inp, lthree_term=True), external_field=field)
    vj.solve_free_boundary_multigrid(replace(inp, lthree_term=True), external_field=field,
                                     raise_on_max_iterations=False)
    assert calls[-1][1]["config"].labelling == "gauge"


@needs_vc
def test_rows_are_the_three_interface_conditions():
    """Normal row = B_out.n/|B|, sheet rows tangent, and together they are the whole jump."""
    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 11), lfreeb=False)
    eq = opt.solve_equilibrium(inp)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    rows, weights = fbt.boundary_residual(inp, eq.solution, field, runtime=eq.solver_context, nphi=16, ntheta=16)
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
    summary = fbt.summarize_boundary_residual(rows, weights)
    assert 0 < summary.normal < 0.05 and 0 < summary.sheet_current < 0.05


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
    out = fbt._levenberg_marquardt(evaluate, x0, rows(x0), "aux", J, ftol=1e-12, max_nfev=50, verbose=1)
    assert out["converged"] and out["reason"] == "ftol" and out["accepted"] >= 1
    np.testing.assert_allclose(out["x"], target, atol=1e-5)
    # a step that fails to solve shrinks the radius, a budget of one evaluation stops at once
    failed = fbt._levenberg_marquardt(lambda x: None, x0, rows(x0), "aux", J, ftol=1e-12, max_nfev=30,
                                      verbose=0, radius=10.0)
    assert not failed["converged"] and failed["reason"] in ("radius", "stalled") and failed["accepted"] == 0
    np.testing.assert_array_equal(failed["x"], x0)
    assert fbt._levenberg_marquardt(evaluate, x0, rows(x0), "aux", J, ftol=1e-12, max_nfev=1,
                                    verbose=0)["reason"] == "max_nfev"
    # a wrong Jacobian: steps keep gaining little, so the finish stops as stalled before its budget
    stale = fbt._levenberg_marquardt(evaluate, x0, rows(x0), "aux", np.vstack([np.eye(3), np.ones(3)]) * 40,
                                     ftol=1e-14, max_nfev=200, verbose=0)
    assert stale["reason"] in ("stalled", "radius") and stale["nfev"] < 200


def test_coil_net_current_is_the_circulation_around_the_axis():
    """G of a purely toroidal 1/R field is its R B_phi, whatever the axis loop it is taken around."""

    class Toroidal:
        def b_cyl(self, r, phi, z):
            return 0.0 * r, 0.7 / r, 0.0 * z

    wout = SimpleNamespace(raxis_cc=np.array([1.0, 0.1]), zaxis_cs=np.array([0.0, -0.05]), nfp=2)
    # 720 points and a second-order tangent: 1.2e-5 relative on this axis
    assert fbt._coil_net_current(Toroidal(), wout) == pytest.approx(0.7, rel=1e-4)


def test_levenberg_marquardt_rejects_a_bad_jacobian():
    """A nonfinite boundary Jacobian, or one whose SVD fails, is a RuntimeError (a rejected trial): the
    LinAlgError it used to raise ended a campaign run (QI 2.5%, A, step 240)."""
    evaluate = lambda z: (np.zeros(2), None)  # noqa: E731
    with pytest.raises(RuntimeError, match="nonfinite"):
        fbt._levenberg_marquardt(evaluate, np.zeros(2), np.ones(2), None, np.array([[1.0, np.nan], [0.0, 1.0]]),
                                 ftol=1e-4, max_nfev=5, verbose=0)


def test_levenberg_marquardt_projects_stiff_coordinates():
    """A stiff closure coordinate (x1 in 100 (x1 - x0^2)) stalls the fixed-Jacobian steps of x0; re-solving it
    after every step (project) reaches the minimum in a few evaluations."""
    def evaluate(x):
        return np.array([x[0] - 1.0, 100.0 * (x[1] - x[0] ** 2)]), None

    x0 = np.zeros(2)
    J = np.array([[1.0, 0.0], [0.0, 100.0]])  # at x0
    plain = fbt._levenberg_marquardt(evaluate, x0, evaluate(x0)[0], None, J, ftol=1e-12, max_nfev=8, verbose=0)
    projected = fbt._levenberg_marquardt(evaluate, x0, evaluate(x0)[0], None, J, ftol=1e-12, max_nfev=8, verbose=0,
                                         project=np.array([1]))
    assert projected["converged"] and np.allclose(projected["x"], 1.0, atol=1e-6)
    assert not plain["converged"] and abs(plain["x"][0] - 1.0) > 1e-2


def test_fit_certificate_measures_stationarity_and_rank():
    """The certificate's stationarity is the Gauss-Newton step's predicted relative gain; a repeated column is a
    rank-deficient Jacobian."""
    J = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    assert fbt._fit_certificate(J, np.array([0.0, 0.0, 2.0]))["stationarity"] == pytest.approx(0.0)
    assert fbt._fit_certificate(J, np.array([1.0, 0.0, 1.0]))["stationarity"] == pytest.approx(0.5)
    flat = fbt._fit_certificate(np.diag([1.0, 1e-3, 0.0])[:, :2], np.array([1e-6, 1e-6, 1.0]))
    assert flat["stationarity"] < 1e-11 and flat["step"] == pytest.approx(1e-3)  # flat: tiny gain, long step
    certificate = fbt._fit_certificate(np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]), np.ones(3))
    assert certificate["rank_deficient"] and certificate["condition"] > 1e12


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
    out = fbt._levenberg_marquardt(evaluate, x0, r0, None, jacobian(x0), ftol=1e-12, max_nfev=40, verbose=0,
                                   project=np.array([1]))
    assert out["converged"] and out["reason"] == "ftol"
    np.testing.assert_allclose(out["x"], [0.75, 0.5625], rtol=1e-6)
    assert out["projections"] == 0
    joint = fbt._levenberg_marquardt(evaluate, x0, r0, None, jacobian(x0), ftol=1e-12, max_nfev=40, verbose=0)
    assert not joint["converged"] and abs(joint["x"][0] - 0.75) > 1e-2


def test_project_stops_at_conditional_stationarity():
    evaluate, jacobian = _stiff_closure()
    x = np.array([0.75, 0.0])
    Q, R = np.linalg.qr(jacobian(x)[:, [1]])
    xs, got, used, stationary = fbt._project(evaluate, x, evaluate(x), Q, R, np.array([1]), 10, 1e-20)
    assert stationary and used == 1  # linear in c: one correction
    np.testing.assert_allclose(xs, [0.75, 0.5625], rtol=1e-14)
    _, _, used, stationary = fbt._project(evaluate, xs, got, Q, R, np.array([1]), 10, 1e-20)
    assert stationary and used == 0
    assert fbt._projection_gate(1e-4) == pytest.approx(1e-5)


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
    stale = fbt._levenberg_marquardt(evaluate, x0, r0, None, jacobian(x0), ftol=1e-10, max_nfev=60, verbose=0,
                                     project=np.array([1]))
    assert stale["projection_certified"] is None and conditional_gain(stale["x"]) > 1e-6
    fresh = fbt._levenberg_marquardt(evaluate, x0, r0, None, jacobian(x0), ftol=1e-10, max_nfev=60, verbose=0,
                                     project=np.array([1]), current_columns=lambda x, aux: (jacobian(x)[:, [1]], aux))
    assert fresh["projection_certified"] is True and conditional_gain(fresh["x"]) <= 1e-11


def test_projection_unconverged_is_not_converged():
    """When the fresh-column corrections cannot reach conditional stationarity the finish says so."""
    evaluate, jacobian, _ = _rotating_current()
    x0 = np.array([2.0, 4.0])
    r0 = evaluate(x0)[0]
    out = fbt._levenberg_marquardt(evaluate, x0, r0, None, jacobian(x0), ftol=1e-10, max_nfev=60, verbose=0,
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
        fbt._levenberg_marquardt(evaluate, x0, evaluate(x0)[0], None, J, ftol=1e-10, max_nfev=10, verbose=0,
                                 project=np.array([1]))


def test_steps_that_each_gain_little_are_a_stall():
    """A linear problem with a large irreducible residual: every step is predicted exactly (gain ratio 1) but gains
    far under STALL_GAIN of the cost, so STALL_STEPS of them in a row stop the steps as stalled."""
    def evaluate(x):
        return np.array([x[0] - 1.0, x[1] - 2.0, 100.0]), None

    J = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    x0 = np.array([3.0, -1.0])
    out = fbt._levenberg_marquardt(evaluate, x0, evaluate(x0)[0], None, J, ftol=1e-12, max_nfev=60, verbose=0,
                                   radius=1e-3)
    assert not out["converged"] and out["reason"] == "stalled"


def test_broyden_update_cuts_the_evaluations_of_a_stale_jacobian():
    """From a stale Jacobian, secant (Broyden) updates along the accepted steps reach the root in far fewer
    evaluations than the fixed Jacobian."""
    def evaluate(x):
        return np.array([x[0] ** 2 - 2.0, x[1] ** 3 - 3.0, x[0] * x[1] - np.sqrt(2) * 3 ** (1 / 3)]), None

    x0 = np.array([3.0, 3.0])
    J0 = np.array([[2 * x0[0], 0.0], [0.0, 3 * x0[1] ** 2], [x0[1], x0[0]]])
    runs = {}
    for mode in ("none", "broyden"):
        runs[mode] = fbt._levenberg_marquardt(evaluate, x0, evaluate(x0)[0], None, J0, ftol=1e-14, max_nfev=200,
                                              verbose=0, update=None if mode == "none" else mode)
        np.testing.assert_allclose(runs[mode]["x"], [np.sqrt(2), 3 ** (1 / 3)], rtol=1e-10)
    assert runs["broyden"]["jacobian_updates"] > 0 and runs["none"]["jacobian_updates"] == 0
    assert runs["broyden"]["nfev"] * 3 < runs["none"]["nfev"]


@needs_vc
@pytest.mark.full
def test_uncertified_tangent_solve_is_no_jacobian(monkeypatch):
    """A boundary Jacobian whose tangent solves are not all certified raises instead of being used."""
    from vmex.core import implicit as im

    inp, field = _small_vacuum()
    model = fbt.ThreeTermFreeBoundaryModel(replace(inp, lfreeb=False), **SMALL)
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
def test_boundary_fit_without_an_equilibrium_raises_runtime_error(monkeypatch):
    """A boundary fit whose Jacobian point has no certified equilibrium raises RuntimeError (a rejected trial),
    not KeyError (which ended a campaign run), and never linearizes another point's state."""
    inp, field = _small_vacuum()
    model = fbt.ThreeTermFreeBoundaryModel(replace(inp, lfreeb=False), **SMALL)
    monkeypatch.setattr(model, "evaluate", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="no certified equilibrium"):
        model.solve_boundary(model.params0, field)


@needs_vc
@pytest.mark.full
def test_interior_state_does_not_depend_on_its_start():
    """At one boundary, the equilibrium solved cold, warm from a nearby boundary's equilibrium, and from a
    perturbed seed gives the same interface rows and boundary Jacobian: no history dependence through the
    interior's m = 1 gauge or the hot restart."""
    inp, field = _small_vacuum()
    model = fbt.ThreeTermFreeBoundaryModel(replace(inp, lfreeb=False), **SMALL)
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
def test_normal_operator_is_the_surface_normal_displacement():
    """N @ dx is the normal displacement of the boundary surface for a coordinate change dx (finite differences of
    the surface against its numerical normal, periodic central differences), a the RMS minor radius."""
    inp, field = _small_vacuum()
    model = fbt.ThreeTermFreeBoundaryModel(replace(inp, lfreeb=False), **SMALL)
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
    np.testing.assert_allclose((N @ dx).reshape(48, 48)[:, 1:-1], fd,
                               atol=2e-2 * np.max(np.abs(fd)))


@needs_vc
@pytest.mark.full
def test_vacuum_free_boundary_agrees_with_nestor():
    """In vacuum the free boundary is a flux surface of the coil field alone: both methods find it.

    VMEC + NESTOR itself lands ~1.6 mm from the exact free boundary of an
    essentially exact field (VMEC2000 1.3 mm), so the methods are compared at
    that level; from either start (the deck, and NESTOR's result through
    ``restart_from``) the interface conditions end below NESTOR's.  At mpol = 4
    the least-squares minimum (labelling "none") is also flat to ~3 mm, so the
    surfaces are compared within the sum of the two floors, and more sharply
    through the residuals and the rotational transform.
    """
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 4, 25), phiedge=-0.025, curtor=0.0,
                  am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    free = vj.solve_free_boundary_multigrid(inp, external_field=field)
    nestor_wout = _wout(inp, free)
    before = fbt.summarize_boundary_residual(*fbt.boundary_residual(inp, free.state, field, nphi=48, ntheta=48))
    # The net-current row's reference: outside the plasma the toroidal circulation is the coils' alone, so a
    # free boundary's edge R B_phi equals the coil current that _coil_net_current integrates on the axis.
    np.testing.assert_allclose(abs(fbt._coil_net_current(field, nestor_wout)), abs(float(nestor_wout.rbtor)),
                               rtol=1e-3)
    config = vj.ThreeTermConfig(labelling="none")
    fits = [vj.solve_free_boundary_three_term(inp, external_field=field, restart_from=start, config=config)
            for start in (None, free)]
    for fit in fits:
        after = _residual(fit.three_term.input, _equilibrium(fit), field)
        assert after.normal < before.normal and after.sheet_current < before.sheet_current
        assert fit.converged and fit.three_term.certified and "interface residuals" in fit.three_term.message
        assert fit.vacuum is None
    strict = vj.solve_free_boundary_three_term(inp, external_field=field, previous=fits[0],
                                               config=replace(config, max_residual=1e-6))
    assert strict.three_term.certified and not strict.converged and "above the accepted" in strict.three_term.message
    walls = [_wout(fit.three_term.input, fit) for fit in fits]
    for wall in walls:   # budget: NESTOR's own floor (~1.6 mm) plus the flat least-squares minimum (~3 mm)
        assert _lcfs_distance(wall, nestor_wout) < 5e-3
    np.testing.assert_allclose(np.asarray(walls[0].iotaf)[[0, -1]], np.asarray(nestor_wout.iotaf)[[0, -1]], atol=3e-3)
    # The model's Jacobian (implicit state tangents, field as an argument) is upstream's implicit Jacobian of
    # the same rows at the same state (the seed's: two independent re-solves of a boundary differ at the
    # floor); and a warm restart at an unchanged field stays where it is.
    fit, model = fits[0], fits[0].three_term.model
    problem = opt.make_problem(model.fixed, objective_terms=[(lambda state, runtime: model.rows_at(
        state, runtime, field), 0.0, 1.0)], weight_semantics="residual", max_mode=model.max_mode,
        vary_major_radius=True, jacobian_batch_size=8, device=jax.devices()[0])
    J = model.linearize(model.x0, (*model.seed, model.params0, True), field)[0]
    J_upstream = np.asarray(problem.residual_jac(model.x0))
    assert np.linalg.norm(J - J_upstream) < 1e-9 * np.linalg.norm(J_upstream)
    again = vj.solve_free_boundary_three_term(inp, external_field=field, previous=fit, config=config)
    # no new Jacobian: the warm start's own certification reuses the latest linearization (counted once)
    assert again.three_term.njev <= 1 and again.converged and _cost(again, field) <= _cost(fit, field) * (1 + 1e-6)
    assert _lcfs_distance(_wout(again.three_term.input, again), walls[0]) < 1e-4


@needs_vc
@pytest.mark.full
def test_finite_beta_free_boundary_removes_the_sheet_current():
    """LP QA at 0.5% beta with its fitted coils, through the free-boundary ladder's ``three_term`` keyword (the
    gauge labelling, the default)."""
    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    design = _residual(inp, opt.solve_equilibrium(replace(inp, lfreeb=False)), field)
    result = vj.solve_free_boundary_multigrid(inp, external_field=field, three_term=vj.ThreeTermConfig())
    assert result.converged and result.vacuum is None and result.three_term.labelling == "gauge"
    res = result.three_term.boundary_residual
    for name in ("normal", "pressure", "sheet_current"):
        assert getattr(res, name) < 0.25 * getattr(design, name), name
    assert res.sheet_current < 1.5e-3


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
    """LP QA at 0.5% beta: boundary and Redl current in one fit of the model (labelling "none"), certified, with
    exact columns for both."""
    from vmex.core import bootstrap as bs

    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    profiles = _lp_beta0p5_profiles(inp)
    model = fbt.ThreeTermFreeBoundaryModel(inp, labelling="none", bootstrap=profiles)
    fit = model.solve_boundary(model.params0, field, ftol=1e-4, jacobian_ftol=1e-4, refresh_on_stall=True)
    fit = model.certify(model.params0, field, fit, gate=fbt.POLISH_FTOL)
    state, mask, params, _ = fit["aux"]
    x = fit["x"]
    assert x.size == model.n_boundary + 3 + 24
    assert model.bootstrap_residual(state, params) < 1e-2
    assert fit["boundary_residual"].sheet_current < 2e-3
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
    J = model.linearize(x, fit["aux"], field)[0]
    for k in (0, model.n_boundary + 3, model.n_boundary + 15):
        h = 1e-5 * model.x_scale[k]
        rows = [model.evaluate(x + sign * h * np.eye(x.size)[k], params, field, tight=True)[0]
                for sign in (1, -1)]
        fd = (rows[0] - rows[1]) / (2 * h)
        assert np.linalg.norm(J[:, k] - fd) < 1e-4 * np.linalg.norm(fd), k


def test_fixed_boundary_bootstrap_model_matches_picard():
    """fixed_boundary=True: the Gauss-Newton current at a fixed boundary is the Picard loop's (both on VMEC2000's
    released m = 1 complement), with exact columns."""
    from vmex.core import bootstrap as bs

    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 25), lfreeb=False,
                  ftol_array=np.array([1e-13]), niter_array=np.array([20000]))
    profiles = _lp_beta0p5_profiles(inp)
    model = fbt.ThreeTermFreeBoundaryModel(inp, labelling="none", bootstrap=profiles, fixed_boundary=True,
                                           m1_complement="frozen")
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
    and the drift of field lines started on it measure that without virtual casing.  From the deck boundary alone
    the least squares ends in a minimum ~4 cm away, so the solve starts from NESTOR's free boundary
    (``restart_from``), on the tricubic mgrid field; there its coil B.n is below VMEC2000's and field lines drift
    under 1 mm from it (at 5 x 4 modes neither boundary is a flux surface to better than ~0.5 mm).  Finite beta:
    field lines need the plasma field too, so the check is the interface residuals: VMEC2000's free boundary keeps
    an edge sheet current (K = 2.4e-3 |B| at 5 x 4 modes), which the three-term boundary reduces.  About 7 minutes
    per case on one GPU."""
    from vmex.core.mgrid import MgridField

    reference = json.loads(CTH_VMEC2000.read_text())[case]
    inp = replace(vj.VmecInput.from_file(DATA / "input.cth_like_free_bdy"), mgrid_file=str(CTH_MGRID))
    if case == "vacuum":
        inp = replace(inp, pres_scale=0.0, curtor=0.0)
    nestor = vj.solve_free_boundary_multigrid(inp, raise_on_max_iterations=False)
    three = vj.solve_free_boundary_multigrid(inp, restart_from=nestor,
                                             three_term=vj.ThreeTermConfig(labelling="none"))
    assert three.converged
    w = _wout(three.three_term.input, three)
    residual = three.three_term.boundary_residual
    vmec2000 = SimpleNamespace(nfp=int(w.nfp), xm=np.asarray(reference["xm"]), xn=np.asarray(reference["xn"]),
                               rmnc=np.asarray([reference["rmnc"]]), zmns=np.asarray([reference["zmns"]]))
    print(f"{case}: three-term LCFS {1e3 * _lcfs_distance(w, vmec2000):.2f} mm from VMEC2000's; residuals "
          f"{residual} against VMEC2000's {reference['residual']}")
    for name in ("normal", "sheet_current"):
        assert getattr(residual, name) < reference["residual"][name]
    if case == "vacuum":
        field = MgridField.from_input(inp, CTH_MGRID, order=3)  # the smooth field the three-term solve uses
        (bn, drift), (bn_ref, drift_ref) = _vacuum_flux_surface(w, field), _vacuum_flux_surface(vmec2000, field)
        print(f"vacuum: coil B.n {bn:.2e} (VMEC2000 {bn_ref:.2e}), field-line drift {1e3 * drift:.2f} mm "
              f"(VMEC2000 {1e3 * drift_ref:.2f} mm)")
        assert bn < 5e-4 and bn < bn_ref
        assert drift < 1e-3  # absolute: at 5 x 4 modes VMEC2000's boundary drifts a little less (see above)


@needs_vc
@pytest.mark.full
def test_secant_fit_reaches_the_trust_region_fit_with_fewer_jacobians(monkeypatch):
    """The cold fit (``ThreeTermConfig.fit``) changes the path, not the certified free boundary: secant rounds
    (Broyden-updated Jacobians) and scipy's trust region (a fresh Jacobian per step) certify the same root, the
    secant one with fewer Jacobians; and secant rounds that cannot converge hand over to the trust region instead
    of failing."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 4, 25), phiedge=-0.025, curtor=0.0,
                  am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    fits = {fit: vj.solve_free_boundary_three_term(inp, external_field=field,
                                                   config=vj.ThreeTermConfig(labelling="none", fit=fit))
            for fit in ("trf", "secant")}
    for fit in fits.values():
        assert fit.converged
    trf, secant = fits["trf"], fits["secant"]
    assert secant.three_term.njev < trf.three_term.njev
    np.testing.assert_allclose(_cost(secant, field), _cost(trf, field), rtol=1e-3)
    assert _lcfs_distance(_wout(inp, secant), _wout(inp, trf)) < 1e-4
    monkeypatch.setattr(fbt, "FIT_ROUNDS", 1)
    monkeypatch.setattr(fbt, "_levenberg_marquardt", _unconverged(fbt._levenberg_marquardt))
    fallback = vj.solve_free_boundary_three_term(inp, external_field=field,
                                                 config=vj.ThreeTermConfig(labelling="none"))
    assert fallback.converged and fallback.three_term.model.fit_fallbacks >= 1
    assert _lcfs_distance(_wout(inp, fallback), _wout(inp, trf)) < 1e-4


def _unconverged(levenberg_marquardt):
    """The secant rounds' Levenberg-Marquardt (``update="broyden"``) stops after one step, unconverged."""
    def wrapped(*args, update=None, **kwargs):
        if update != "broyden":
            return levenberg_marquardt(*args, update=update, **kwargs)
        out = levenberg_marquardt(*args, update=update, **dict(kwargs, max_nfev=2))
        return dict(out, converged=False, reason="max_nfev")
    return wrapped


@needs_vc
@pytest.mark.full
def test_certify_polishes_on_tight_states(monkeypatch):
    """The polish of a fit that failed its certificate evaluates tight states (the ones the certificate judges),
    while the fit itself stays on trial states (labelling "none")."""
    inp, field = _small_vacuum()
    calls, original, real = [], fbt.ThreeTermFreeBoundaryModel.solve_boundary, fbt._fit_certificate
    count = {"n": 0}

    def spy(self, *args, **kwargs):
        calls.append(bool(kwargs.get("tight", False)))
        return original(self, *args, **kwargs)

    def first_fails(J, r, *args):
        count["n"] += 1
        certificate = real(J, r, *args)
        return dict(certificate, stationarity=1.0) if count["n"] == 1 else certificate

    monkeypatch.setattr(fbt.ThreeTermFreeBoundaryModel, "solve_boundary", spy)
    monkeypatch.setattr(fbt, "_fit_certificate", first_fails)
    result = vj.solve_free_boundary_three_term(inp, external_field=field,
                                               config=vj.ThreeTermConfig(labelling="none", **SMALL))
    assert result.three_term.certified
    assert calls[0] is False and True in calls[1:]


@needs_vc
@pytest.mark.full
def test_an_exhausted_polish_budget_is_named(monkeypatch):
    """A fit still short of stationarity when the polish rounds run out is not certified, with the budget named
    (so a record can be classified as budget-exhausted rather than as a fit with no stationary point); a finish
    that only stalled is not converged either."""
    inp, field = _small_vacuum()
    real = fbt._fit_certificate
    monkeypatch.setattr(fbt, "_fit_certificate", lambda J, r, *args: dict(real(J, r, *args), stationarity=1.0))
    monkeypatch.setattr(fbt, "POLISH_ROUNDS", 0)
    config = vj.ThreeTermConfig(labelling="none", **SMALL)
    result = vj.solve_free_boundary_three_term(inp, external_field=field, config=config)
    assert not result.converged and not result.three_term.certified and result.three_term.certificate is None
    assert "polish budget of 0 rounds ran out" in result.three_term.message
    monkeypatch.undo()

    def stall(evaluate, x, r, aux, J, **kwargs):
        return dict(x=x, rows=r, aux=aux, nfev=1, accepted=0, converged=False, radius=kwargs.get("radius"),
                    reason="stalled", projections=0)

    monkeypatch.setattr(fbt, "_levenberg_marquardt", stall)
    stalled = vj.solve_free_boundary_three_term(inp, external_field=field, config=replace(config, max_nfev=2))
    assert not stalled.converged and "not certified" in stalled.three_term.message
    assert stalled.three_term.certificate is None
