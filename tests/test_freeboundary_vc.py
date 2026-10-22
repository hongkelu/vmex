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
def test_condensed_labelling_with_the_automatic_weight():
    """``labelling="condensed"`` with VMEC's automatic weight: the weight is set at the first fit and the fit
    still reduces B.n and the sheet current of the deck's fixed-boundary equilibrium."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 4, 7), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    design = _residual(inp, opt.solve_equilibrium(replace(inp, lfreeb=False)), field, n=12)
    result = fvc.solve_free_boundary_three_term(inp, external_field=field, labelling="condensed",
                                                label_weight="auto", **dict(SMALL, max_mode=3), max_nfev=20)
    res = result.boundary_residual
    assert 0 < result.model.label_weight < np.inf and not result.model._label_auto
    assert res.normal < design.normal and res.sheet_current < design.sheet_current


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
    assert problem.equilibrium_from_x(x).wout.aspect == pytest.approx(result.fun, rel=1e-6)
    problem.close()
    with pytest.raises(ValueError, match="evaluate the candidate"):
        problem.accept_x(x0 + 0.123)
    assert problem.stats["trials"] >= 3 and problem.stats["linearizations"] >= 1


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
    assert again.njev == 0 and again.cost <= fit.cost * (1 + 1e-6)
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
