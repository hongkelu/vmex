"""Free boundary from the three virtual-casing interface conditions.

:mod:`vmex.core.freeboundary_vc` solves for the boundary on which ``B.n``,
the pressure jump and the sheet current vanish. Lanes: input validation; the
residual rows' definitions on a converged finite-beta state; and (``full``)
the vacuum and finite-beta Landreman--Paul QA cases with their bundled ESSOS
coils, where the solve agrees with NESTOR and is independent of its start.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
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
        fvc.solve_free_boundary_three_terms(inp, external_field=lambda xyz: xyz)


def test_asymmetric_boundary_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), lasym=True)
    with pytest.raises(NotImplementedError, match="lasym"):
        fvc.solve_free_boundary_three_terms(inp, external_field=lambda xyz: xyz)


def test_unknown_boundary_condition_is_rejected():
    with pytest.raises(ValueError, match="boundary_condition"):
        vj.solve_free_boundary_multigrid(_deck("input.LandremanPaul2021_QA_lowres", 3, 11),
                                         external_field=lambda xyz: xyz, boundary_condition="sheet")


@needs_vc
@pytest.mark.usefixtures("_module_jit_enabled")
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
    """The three conditions on ``eq``'s boundary, with the fixed singular quadrature (4 nfp n, 2 n) that the
    model's docstring recommends (planning it to 4 digits can exhaust a GPU on a finite-beta surface)."""
    data = vc.surface_field_data_from_state(inp, eq.solution, runtime=eq.solver_context, nphi=n, ntheta=n)
    precision = vc.plan_vc_precision(data, quad_nt=4 * int(inp.nfp) * n, quad_np=2 * n)
    return fvc.summarize_boundary_residual(*fvc.boundary_residual(
        inp, eq.solution, field, runtime=eq.solver_context, nphi=n, ntheta=n, precision=precision))


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
@pytest.mark.usefixtures("_module_jit_enabled")
def test_vacuum_free_boundary_agrees_with_nestor():
    """In vacuum the free boundary is a flux surface of the coil field alone: both methods find it.

    The answer depends on the deck: its boundary is the reference of the poloidal labelling (module
    docstring), not only the start.  Here the deck is the one VMEC + NESTOR leaves; from it and from the
    original LP deck's boundary the solve lands on one boundary, with the interface conditions below NESTOR's.
    With the original LP deck as the deck the answer differs (B.n 9.7e-4) and is again start-independent.  VMEC + NESTOR itself lands ~1.6 mm from the exact free boundary of an
    essentially exact field (VMEC2000 1.3 mm), so the surfaces are compared at that level, and more sharply
    through the residuals and the rotational transform.  At mpol = 4 the labelling matters: with the original
    LP deck's, truncated to 4 modes, B.n ends at 9.7e-4, with NESTOR's at 5.4e-4 (NESTOR's own: 5.9e-4).
    """
    lp = replace(_deck("input.LandremanPaul2021_QA_lowres", 4, 25), phiedge=-0.025, curtor=0.0,
                 am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    nestor, nestor_deck, reported = _nestor(lp, field)
    inp = replace(lp, rbc=nestor_deck.rbc, zbs=nestor_deck.zbs, raxis_c=nestor_deck.raxis_c,
                  zaxis_s=nestor_deck.zaxis_s)
    before = _residual(inp, nestor, field)
    for name in ("normal", "pressure", "sheet_current"):   # the gate on the NESTOR result
        np.testing.assert_allclose(getattr(reported, name), getattr(before, name), rtol=0.25)
    # The net-current row's reference: outside the plasma the toroidal circulation is the coils' alone, so a
    # free boundary's edge R B_phi equals the coil current that _coil_net_current integrates on the axis.
    np.testing.assert_allclose(abs(fvc._coil_net_current(field, nestor.wout)), abs(float(nestor.wout.rbtor)),
                               rtol=1e-3)
    fits = [fvc.solve_free_boundary_three_terms(inp, external_field=field, initial_boundary=start)
            for start in (None, lp)]
    np.testing.assert_allclose(fits[1].x, fits[0].x, rtol=0, atol=1e-9)
    for fit in fits:
        after = _residual(inp, fit.equilibrium, field)
        assert fit.success and after.normal < before.normal and after.sheet_current < before.sheet_current
    walls = [fit.equilibrium.wout for fit in fits]
    for wall in walls:   # budget: NESTOR's own floor (~1.6 mm)
        assert _lcfs_distance(wall, nestor.wout) < 2e-3
    np.testing.assert_allclose(np.asarray(walls[0].iotaf)[[0, -1]], np.asarray(nestor.wout.iotaf)[[0, -1]], atol=3e-3)
    # With the LP deck itself as the deck (its labelling, truncated to 4 modes, as the reference) the answer is
    # another one (B.n 9.7e-4), but again the same from either start.
    lp_fits = [fvc.solve_free_boundary_three_terms(lp, external_field=field, initial_boundary=start)
               for start in (None, nestor_deck)]
    np.testing.assert_allclose(lp_fits[1].x, lp_fits[0].x, rtol=0, atol=1e-9)
    assert all(fit.success for fit in lp_fits)
    # a warm restart at an unchanged field stays where it is.
    fit = fits[0]
    again = fvc.solve_free_boundary_three_terms(inp, external_field=field, previous=fit)
    assert again.njev == 0 and again.cost <= max(fit.cost, fit.model._NEWTON_COST)
    assert _lcfs_distance(again.equilibrium.wout, walls[0]) < 1e-4


_FINITE_BETA_LEAK = pytest.mark.xfail(strict=True, reason=(
    "known limitation: the finite-beta rows are B.n moments on the boundary's modes (two below the interior's), "
    "and B.n outside them is not controlled; LP QA 0.5 % beta with its bundled coils at mpol 4 ends at "
    "B.n 5.9e-3 (deck 6.0e-3), at mpol 6 1.18e-3 (deck 1.17e-3); see docs/explanation/validation.md"))


@needs_vc
@pytest.mark.full
@_FINITE_BETA_LEAK
@pytest.mark.usefixtures("_module_jit_enabled")
def test_finite_beta_free_boundary_removes_the_sheet_current():
    """LP QA at 0.5% beta with its fitted coils, through the solver's public option."""
    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    design = _residual(inp, opt.solve_equilibrium(replace(inp, lfreeb=False)), field)
    result = vj.solve_free_boundary_multigrid(inp, external_field=field, boundary_condition="three_terms",
                                              three_terms_options=dict(quadrature=(4 * int(inp.nfp) * 48, 96)))
    assert result.converged and result.vacuum is None
    res = result.boundary_residual
    for name in ("normal", "pressure", "sheet_current"):
        assert getattr(res, name) < 0.25 * getattr(design, name), name
    assert res.sheet_current < 1.5e-3


@needs_vc
@pytest.mark.usefixtures("_module_jit_enabled")
def test_vacuum_free_boundary_is_the_least_squares_boundary_from_any_start(monkeypatch):
    """Vacuum, LP QA at mpol 3: the rows are the h-gradient of the mean square of B_coil . n (exact Hessian as
    the Jacobian), Newton reaches them from the deck and from another start, and both land on one boundary."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    start = replace(inp, rbc=np.asarray(inp.rbc) * 1.01, zbs=np.asarray(inp.zbs) * 0.99)
    fits = [fvc.solve_free_boundary_three_terms(inp, external_field=field, nphi=24, ntheta=24, initial_boundary=b)
            for b in (None, start)]
    model = fits[0].model
    for fit in fits:
        assert fit.success and fit.cost < 1e-18
        assert fit.boundary_residual.normal < fit.initial_boundary_residual.normal
    np.testing.assert_allclose(fits[1].x, fits[0].x, rtol=0, atol=1e-10)
    # the rows at the deck: R0 d/dh of the mean square of B.n / B0, against central differences of it
    rows = np.asarray(model._rows(model.seed[0], model.params0, field))
    nfp, ntor = int(inp.nfp), int(inp.ntor)
    n, m = np.meshgrid(np.arange(-ntor, ntor + 1) * nfp, np.arange(inp.mpol), indexing="ij")
    phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi / nfp, 24, endpoint=False),
                             np.linspace(0, 2 * np.pi, 24, endpoint=False), indexing="ij")

    def mean_square(x):
        p = model.with_boundary(model.params0, jnp.asarray(x))
        return float(fvc._mean_square_normal_field(jnp.ravel(p.rbc), jnp.ravel(p.zbs), jnp.asarray(m.ravel(), float),
                                                   jnp.asarray(n.ravel(), float), phi, theta, jnp.asarray(phi[:, 0]),
                                                   field))

    R0, eye = float(np.asarray(inp.rbc)[ntor, 0]), np.eye(model.x0.size)
    for k in (0, 3):
        fd = (mean_square(1e-6 * eye[k]) - mean_square(-1e-6 * eye[k])) / 2e-6 * R0 / (2 * model.B0**2)
        np.testing.assert_allclose(rows[k], fd, rtol=1e-5)
    # the Jacobian (exact Hessian through the implicit state tangents) against re-solved central differences
    J = model.linearize(fits[0].x, fits[0].aux, field)[0]
    k, h = 2, 1e-6
    fd = [model.evaluate(fits[0].x + s * h * eye[k], model.params0, field, tight=True)[0] for s in (1, -1)]
    np.testing.assert_allclose(J[:, k], (fd[0] - fd[1]) / (2 * h), rtol=1e-4, atol=1e-6 * np.abs(J[:, k]).max())
    again = fvc.solve_free_boundary_three_terms(inp, external_field=field, previous=fits[0])
    assert again.njev == 0 and again.success
    assert 0.0 < fits[0].normal_outside_modes <= fits[0].boundary_residual.normal
    # a solve that does not lower B.n says so, with the part outside the boundary's modes
    monkeypatch.setattr(model, "boundary_residual", lambda *args: fvc.BoundaryResidual(1.0, 1.0, 1.0))
    with pytest.warns(RuntimeWarning, match="outside the boundary's"):
        fvc.solve_free_boundary_three_terms(inp, external_field=field, previous=fits[0])


def test_square_galerkin_boundary_is_solved_by_newton_from_a_stale_jacobian():
    """Square rows (the Galerkin free boundary): Newton reaches the root to round-off, reusing a stale Jacobian
    while it converges fast and refreshing it otherwise; the root does not depend on the start."""

    def rows(x):
        return np.array([x[0] ** 2 + x[1] - 3.0, x[0] - x[1] ** 3 + 7.0])

    def jac(x):
        return np.array([[2.0 * x[0], 1.0], [1.0, -3.0 * x[1] ** 2]])

    model = fvc.ThreeTermsFreeBoundaryModel.__new__(fvc.ThreeTermsFreeBoundaryModel)
    model.vacuum, model.fixed_boundary = True, False
    model.x0, model.x_scale = np.zeros(2), np.ones(2)
    model.cfg = model.cfg_trial = None
    model.evaluate = lambda z, params, field, tight=False: (rows(z), (z, None, None, True))
    model.linearize = lambda z, aux, field, extra=None: (jac(z), None, None, (z, None, None, True))
    root = np.array([1.0, 2.0])
    for start, jacobian in ((np.array([1.3, 1.7]), None), (np.array([0.8, 2.2]), jac(np.array([1.5, 1.5])))):
        out = model.solve_boundary(None, None, x0=start, jacobian=jacobian)
        assert out["converged"]
        np.testing.assert_allclose(out["x"], root, atol=1e-11)
        assert out["nfev"] <= 12


def test_newton_stops_at_the_noise_of_the_rows():
    """Rows with evaluation noise (as virtual-casing rows have): below ``_NEWTON_FLOOR`` Newton stops once a step
    no longer converges fast, instead of spending Jacobians on the noise, and reports convergence."""
    noise = 1e-11

    def rows(x):
        draw = np.random.default_rng(np.frombuffer(np.asarray(x, dtype=float).tobytes(), dtype=np.uint32)).normal(size=2)  # a new draw per point
        return np.array([x[0] ** 2 + x[1] - 3.0, x[0] - x[1] ** 3 + 7.0]) + noise * draw

    def jac(x):
        return np.array([[2.0 * x[0], 1.0], [1.0, -3.0 * x[1] ** 2]])

    model = fvc.ThreeTermsFreeBoundaryModel.__new__(fvc.ThreeTermsFreeBoundaryModel)
    model.vacuum, model.fixed_boundary = True, False
    model.x0, model.x_scale = np.zeros(2), np.ones(2)
    model.cfg = model.cfg_trial = None
    model.evaluate = lambda z, params, field, tight=False: (rows(z), (z, None, None, True))
    model.linearize = lambda z, aux, field, extra=None: (jac(z), None, None, (z, None, None, True))
    for jacobian in (None, jac(np.array([1.0, 2.0]))):  # fresh Jacobians; a reused one at the root
        out = model.solve_boundary(None, None, x0=np.array([1.3, 1.7]) if jacobian is None else np.array([1.0, 2.0]),
                                   jacobian=jacobian)
        assert out["converged"] and 0.5 * out["rows"] @ out["rows"] <= model._NEWTON_FLOOR
        np.testing.assert_allclose(out["x"], [1.0, 2.0], atol=1e-9)
        assert out["njev"] <= 6


def test_unknown_problem_boundary_condition_is_rejected():
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    with pytest.raises(ValueError, match="boundary_condition"):
        FreeBoundaryProblem.from_loss(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), lambda s, r, x: 0.0,
                                      np.zeros(1), field_from_parameters=lambda x: None, boundary_condition="sheet")


def test_problem_boundary_condition_dispatches_to_three_terms(monkeypatch):
    """``boundary_condition="three_terms"`` hands the loss, maps, quantities and options to the three-terms problem."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    calls = []
    monkeypatch.setattr(fvc.ThreeTermsFreeBoundaryProblem, "from_loss",
                        classmethod(lambda cls, *args, **kwargs: calls.append((args, kwargs)) or "problem"))
    inp, loss, field = _deck("input.LandremanPaul2021_QA_lowres", 3, 11), (lambda s, r, x: 0.0), (lambda x: None)
    got = FreeBoundaryProblem.from_loss(inp, loss, np.zeros(1), field_from_parameters=field, quantities=(len,),
                                        boundary_condition="three_terms", three_terms_options=dict(nphi=16))
    (args, kwargs), = calls
    assert got == "problem" and args[:2] == (inp, loss) and kwargs["field_from_parameters"] is field
    assert kwargs["quantities"] == (len,) and kwargs["nphi"] == 16


@needs_vc
@pytest.mark.usefixtures("_module_jit_enabled")
def test_three_terms_problem_gradients_match_resolved_differences():
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
        boundary_condition="three_terms",
        three_terms_options=dict(nphi=24, ntheta=24, quadrature=(4 * 2 * 24, 48),
                                boundary_max_nfev=80))
    assert isinstance(problem, fvc.ThreeTermsFreeBoundaryProblem)
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


@needs_vc
@pytest.mark.full
@_FINITE_BETA_LEAK
@pytest.mark.usefixtures("_module_jit_enabled")
def test_bootstrap_current_solved_with_the_free_boundary():
    """LP QA at 0.5% beta: boundary and Redl current in one solve, with exact columns for both."""
    from vmex.core import bootstrap as bs

    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    profiles = _lp_beta0p5_profiles(inp)
    fit = fvc.solve_free_boundary_three_terms(inp, external_field=field, bootstrap=profiles,
                                              quadrature=(4 * int(inp.nfp) * 48, 96))
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


@pytest.mark.usefixtures("_module_jit_enabled")
def test_fixed_boundary_bootstrap_model_matches_picard():
    """fixed_boundary=True: the Gauss-Newton current at a fixed boundary is the Picard loop's, with exact columns."""
    from vmex.core import bootstrap as bs

    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 25), lfreeb=False,
                  ftol_array=np.array([1e-13]), niter_array=np.array([20000]))
    profiles = _lp_beta0p5_profiles(inp)
    model = fvc.ThreeTermsFreeBoundaryModel(inp, bootstrap=profiles, fixed_boundary=True)
    assert model.n_boundary == 0 and model.x0.size == 3 + 24  # three knots inside s_1, then the half mesh
    out = model.solve_boundary(model.params0, None)
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


def test_newton_solves_the_trials_tight_once_their_tolerance_is_the_noise():
    """Trials at a looser force tolerance (``trial_ftol``) carry noise; when a fresh Newton step no longer descends
    on them, the trials are solved to the deck's tolerance and Newton converges to round-off."""

    def rows(x, tight):
        exact = np.array([x[0] ** 2 + x[1] - 3.0, x[0] - x[1] ** 3 + 7.0])
        noise = np.random.default_rng(np.frombuffer(np.asarray(x, dtype=float).tobytes(), dtype=np.uint32)).normal(size=2)  # a new draw per point
        return exact if tight else exact + 1e-6 * noise

    def jac(x):
        return np.array([[2.0 * x[0], 1.0], [1.0, -3.0 * x[1] ** 2]])

    model = fvc.ThreeTermsFreeBoundaryModel.__new__(fvc.ThreeTermsFreeBoundaryModel)
    model.vacuum, model.fixed_boundary = True, False
    model.x0, model.x_scale, model.cfg, model.cfg_trial = np.zeros(2), np.ones(2), "deck", "trial"
    model.evaluate = lambda z, params, field, tight=False: (rows(z, tight), (z, None, None, tight))
    model.linearize = lambda z, aux, field, extra=None: (jac(z), None, None, (z, None, None, True))
    out = model.solve_boundary(None, None, x0=np.array([1.3, 1.7]))
    assert out["converged"] and out["aux"][3]
    np.testing.assert_allclose(out["x"], [1.0, 2.0], atol=1e-11)


def test_newton_backs_off_a_start_without_equilibrium_and_reports_a_stall(capsys):
    """A start with no certified equilibrium backs off toward the deck; rows without a root stall at their
    stationary point, reported as not converged (and printed with ``verbose``)."""

    def rows(x):
        return np.array([x[0] ** 2 + 1.0])  # no root: the cost is stationary at x = 0

    model = fvc.ThreeTermsFreeBoundaryModel.__new__(fvc.ThreeTermsFreeBoundaryModel)
    model.vacuum, model.fixed_boundary = True, False
    model.x0, model.x_scale, model.cfg = np.zeros(1), np.ones(1), None
    model.cfg_trial = model.cfg
    model.evaluate = lambda z, params, field, tight=False: (
        None if abs(z[0]) > 1.0 else (rows(z), (z, None, None, True)))
    model.linearize = lambda z, aux, field, extra=None: (np.array([[2.0 * z[0] + 1e-300]]), None, None, aux)
    out = model.solve_boundary(None, None, x0=np.array([1.2]), verbose=1)  # 1.2 fails, 0.9 = 0.75 x 1.2 is solved
    assert not out["converged"] and abs(out["x"][0]) < 0.9
    assert "Newton stalled" in capsys.readouterr().out
    model.evaluate = lambda z, params, field, tight=False: None
    with pytest.raises(RuntimeError, match="no certified equilibrium"):
        model.solve_boundary(None, None, x0=np.array([1.2]))


def test_fixed_boundary_model_needs_the_bootstrap_current():
    with pytest.raises(ValueError, match="bootstrap"):
        fvc.ThreeTermsFreeBoundaryModel(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), fixed_boundary=True)


def test_coil_net_current_is_the_circulation_around_the_axis():
    """G = (1/2 pi) times the circulation of the field around the magnetic axis: R B_phi for a 1/R field."""
    from types import SimpleNamespace

    @dataclass(frozen=True)
    class Toroidal:
        b0r0: float

        def b_cyl(self, r, phi, z):
            return 0.0 * r, self.b0r0 / r, 0.0 * r

    wout = SimpleNamespace(nfp=2, raxis_cc=np.array([1.0, 0.05]), zaxis_cs=np.array([0.0, 0.03]))
    assert fvc._coil_net_current(Toroidal(1.7), wout) == pytest.approx(1.7, rel=1e-10)


@needs_vc
@pytest.mark.usefixtures("_module_jit_enabled")
def test_finite_beta_rows_are_the_transported_normal_field_moments():
    """Finite beta: the boundary keeps two modes fewer than the interior, and the rows are the area-weighted
    moments of B_out . n / B0 (coils + virtual casing) against R0 / B0 B_out . grad cos(m theta - n nfp phi) on
    its modes, then the net-current row.  B_out's contravariant components here come from the surface metric."""
    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 11)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    nfp, n, ntor = int(inp.nfp), 16, int(inp.ntor)
    model = fvc.ThreeTermsFreeBoundaryModel(inp, nphi=n, ntheta=n, quadrature=(4 * nfp * n, 2 * n))
    assert not model.vacuum and model.max_mode == max(inp.mpol - 1, inp.ntor) - fvc._BOUNDARY_MODE_GAP
    state = model.seed[0]
    rows = np.asarray(model._rows(state, model.params0, field))
    runtime = model._im.runtime_from_params(model.params0, model.cfg)
    data = vc.surface_field_data_from_state(model.fixed, state, runtime=runtime, nphi=n, ntheta=n)
    B_out = np.asarray(vc.PlasmaVacuumInterface.from_surface_data(
        data, p_edge=0.0, precision=model.precision).total_B_out(field))
    normal, area = np.asarray(data.normal), np.linalg.norm(np.asarray(data.area_vector), axis=0)
    bn, weights = np.sum(B_out * normal, axis=0) / model.B0, area / area.sum()
    phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi / nfp, n, endpoint=False),
                             np.linspace(0, 2 * np.pi, n, endpoint=False), indexing="ij")
    rbc, zbs = np.asarray(model.fixed.rbc), np.asarray(model.fixed.zbs)
    R = Rt = Rp = Zt = Zp = 0.0
    for k in range(-ntor, ntor + 1):
        for m in range(int(inp.mpol)):
            a = m * theta - k * nfp * phi
            R, Rt, Rp = R + rbc[k + ntor, m] * np.cos(a), Rt - m * rbc[k + ntor, m] * np.sin(a), \
                Rp + k * nfp * rbc[k + ntor, m] * np.sin(a)
            Zt, Zp = Zt + m * zbs[k + ntor, m] * np.cos(a), Zp - k * nfp * zbs[k + ntor, m] * np.cos(a)
    e_t = np.stack([Rt * np.cos(phi), Rt * np.sin(phi), Zt])
    e_p = np.stack([Rp * np.cos(phi) - R * np.sin(phi), Rp * np.sin(phi) + R * np.cos(phi), Zp])
    g = np.array([[np.sum(e_t * e_t, 0), np.sum(e_t * e_p, 0)], [np.sum(e_t * e_p, 0), np.sum(e_p * e_p, 0)]])
    covariant = np.stack([np.sum(B_out * e_t, 0), np.sum(B_out * e_p, 0)])
    b_t, b_p = np.linalg.solve(np.moveaxis(g, (0, 1), (-2, -1)), np.moveaxis(covariant, 0, -1)[..., None])[..., 0].T
    b_t, b_p = b_t.T, b_p.T
    R0 = float(rbc[ntor, 0])
    expected = [np.sum(weights * bn * R0 / model.B0 * (k * nfp * b_p - m * b_t) * np.sin(m * theta - k * nfp * phi))
                for m, k in model.boundary_modes[:-1]]
    np.testing.assert_allclose(rows[:-1], expected, rtol=1e-8, atol=1e-13)


def test_finite_beta_newton_falls_back_on_the_plain_moments():
    """Finite beta: when Newton on the transported rows stalls (here at a stationary point of the cost that is no
    root), the plain moments are solved first and the transported rows from their answer."""

    def rows(z, moments):
        return np.array([z[0] + 1.7693]) if moments else np.array([z[0] ** 3 - 2.0 * z[0] + 2.0])

    def jac(z, moments):
        return np.array([[1.0]]) if moments else np.array([[3.0 * z[0] ** 2 - 2.0]])

    model = fvc.ThreeTermsFreeBoundaryModel.__new__(fvc.ThreeTermsFreeBoundaryModel)
    model.vacuum, model.fixed_boundary, model._moments = False, False, False
    model.x0, model.x_scale, model.cfg = np.zeros(1), np.ones(1), None
    model.cfg_trial = model.cfg
    model.evaluate = lambda z, params, field, tight=False: (rows(z, model._moments), (z, None, None, True))
    model.linearize = lambda z, aux, field, extra=None: (jac(z, model._moments), None, None, aux)
    out = model.solve_boundary(None, None, x0=np.array([0.0]), verbose=1)
    assert out["converged"] and not model._moments
    np.testing.assert_allclose(out["x"][0] ** 3 - 2.0 * out["x"][0] + 2.0, 0.0, atol=1e-12)
