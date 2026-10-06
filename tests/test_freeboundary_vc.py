"""Free boundary from the three virtual-casing interface conditions.

:mod:`vmex.core.freeboundary_vc` solves for the boundary on which ``B.n``,
the pressure jump and the sheet current vanish. Lanes: input validation; the
residual rows' definitions on a converged finite-beta state; and (``full``)
the vacuum and finite-beta Landreman--Paul QA cases with their bundled ESSOS
coils, where the solve agrees with NESTOR and is independent of its start.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
needs_vc = pytest.mark.skipif(not vc.have_virtual_casing_jax(), reason="requires virtual_casing_jax")


def _deck(name, mpol, ns):
    inp = vj.VmecInput.from_file(DATA / name).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=16)
    return replace(inp, lfreeb=True, mgrid_file="essos_coils(direct)", ns_array=np.array([ns]),
                   niter_array=np.array([5000]), ftol_array=np.array([1e-10]), delt=0.5)


def _coil_field(name):
    """The bundled ESSOS coils as VMEX's exact filament field (no mgrid tabulation)."""
    pytest.importorskip("essos")
    from essos.coils import Coils
    from vmex.core.freeboundary_problem import CoilParameters

    chart = CoilParameters.from_coils(Coils.from_json(str(DATA / name)), current_dofs=())
    return chart(chart.x0)


def test_nonzero_edge_pressure_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), pmass_type="power_series",
                  am=np.array([1.0e3, -0.5e3] + [0.0] * 19))
    with pytest.raises(ValueError, match="edge pressure"):
        fvc.solve_free_boundary_virtual_casing(inp, external_field=lambda xyz: xyz)


def test_asymmetric_boundary_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), lasym=True)
    with pytest.raises(NotImplementedError, match="lasym"):
        fvc.solve_free_boundary_virtual_casing(inp, external_field=lambda xyz: xyz)


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
@pytest.mark.usefixtures("_module_jit_enabled")
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
    fits = [fvc.solve_free_boundary_virtual_casing(inp, external_field=field, initial_boundary=start)
            for start in (None, nestor_deck)]
    for fit in fits:
        after = _residual(inp, fit.equilibrium, field)
        assert after.normal < before.normal and after.sheet_current < before.sheet_current
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
    again = fvc.solve_free_boundary_virtual_casing(inp, external_field=field, previous=fit)
    assert again.njev == 0 and again.cost <= fit.cost * (1 + 1e-6)
    assert _lcfs_distance(again.equilibrium.wout, walls[0]) < 1e-4


@needs_vc
@pytest.mark.full
@pytest.mark.usefixtures("_module_jit_enabled")
def test_finite_beta_free_boundary_removes_the_sheet_current():
    """LP QA at 0.5% beta with its fitted coils, through the solver's public option."""
    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    design = _residual(inp, opt.solve_equilibrium(replace(inp, lfreeb=False)), field)
    result = vj.solve_free_boundary_multigrid(inp, external_field=field, boundary_condition="virtual_casing")
    assert result.converged and result.vacuum is None
    res = result.boundary_residual
    for name in ("normal", "pressure", "sheet_current"):
        assert getattr(res, name) < 0.25 * getattr(design, name), name
    assert res.sheet_current < 1.5e-3
