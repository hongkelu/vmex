"""FreeBoundaryProblem: a loss and constraint rows of coil (and PHIEDGE) designs at the VMEX + NESTOR free boundary.

The problem is a thin layer on ``solve_free_boundary_implicit`` and its adjoint; its values and gradient against
main's anchored implicit gradient are tests/test_freeboundary_problem_seed.py.  Here: the definitions it rejects,
its constraint rows and Jacobian against re-solved central differences, and one SLSQP step through ``minimize``.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from vmex import optimize as opt
from vmex.core.errors import TrialRejected
from vmex.core.freeboundary_problem import FreeBoundaryProblem

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_public_names_share_the_core_classes():
    assert opt.TrialRejected is TrialRejected
    assert opt.FreeBoundaryProblem is FreeBoundaryProblem


def test_from_loss_rejects_invalid_definitions():
    inp = SimpleNamespace(lfreeb=True)
    field = lambda x: x  # noqa: E731
    with pytest.raises(TypeError, match="callable"):
        FreeBoundaryProblem.from_loss(inp, lambda *a: 0., np.zeros(2), field_from_parameters=field,
                                      parameter_quantities=(1,))
    with pytest.raises(TypeError, match="callable"):
        FreeBoundaryProblem.from_loss(inp, lambda *a: 0., np.zeros(2), field_from_parameters=None)
    jax.config.update("jax_enable_x64", True)
    with pytest.raises(ValueError, match="free-boundary"):
        FreeBoundaryProblem.from_loss(SimpleNamespace(lfreeb=False), lambda *a: 0., np.zeros(2),
                                      field_from_parameters=field)
    with pytest.raises(ValueError, match="boundary_condition"):
        FreeBoundaryProblem.from_loss(inp, lambda *a: 0., np.zeros(2), field_from_parameters=field,
                                      boundary_condition="sheet")


@pytest.fixture(scope="module")
def problem():
    """Aspect ratio of the LP QA deck (ns 11, mpol = ntor = 3) in its ESSOS coils: x = (PHIEDGE, coil currents)."""
    jax.config.update("jax_enable_x64", True)
    pytest.importorskip("essos")
    from essos.coils import Coils
    from essos.fields import BiotSavart

    import vmex as vj
    from vmex.core import implicit as im

    coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))
    base = coils.dofs
    n = coils.dofs_curves.size
    inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres").change_resolution(
        mpol=3, ntor=3, ntheta=12, nzeta=16)
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field", phiedge=-0.025, ns_array=np.array([11]),
                  niter_array=np.array([20000]), ftol_array=np.array([1e-12]))
    return FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: im.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: BiotSavart(coils.with_dofs(base.at[n:].multiply(1 + 0.05 * x[1]))),
        plasma_from_parameters=lambda params, x: replace(params, phiedge=params.phiedge * (1 + 0.05 * x[0])),
        quantities=(opt.major_radius,),
        parameter_quantities=(lambda state, runtime, x: im.aspect_ratio(state, runtime) * (1 + x[1]),))


@pytest.mark.full
def test_rows_and_jacobian_match_resolved_differences(problem):
    """Loss gradient and constraint Jacobian against central differences of independent solves, along PHIEDGE
    (a plasma coordinate) and the coil currents (a field coordinate)."""
    x0 = problem.x0
    value, grad = problem.value_and_grad(x0)
    jac = problem.constraint_jac(x0)
    assert jac.shape == (2, 2) and np.all(np.isfinite(jac))
    size = np.abs(np.r_[value, problem.constraint_values(x0)])
    for k in range(2):
        h, e = 1e-3, np.eye(2)[k]
        rows = [np.r_[problem.fun(x0 + s * h * e), problem.constraint_values(x0 + s * h * e)] for s in (1, -1)]
        fd = (rows[0] - rows[1]) / (2 * h)
        # each row against its own size: re-solved differences carry ~1e-6 trial noise over h (measured: the major
        # radius along PHIEDGE, 3.50e-4 against 3.36e-4, 1.3e-5 of a 1 m row)
        assert np.all(np.abs(np.r_[grad[k], jac[:, k]] - fd) <= 2e-3 * np.abs(fd) + 2e-5 * size), (k, grad, jac, fd)


@pytest.mark.full
def test_one_slsqp_step_accepts_an_evaluated_point(problem):
    """minimize starts at the accepted point, promotes the step it takes and stops at the step budget."""
    before = problem.accepted.parameters.copy()
    result = opt.minimize(problem, x0=before, method="SLSQP", options=dict(maxiter=1),
                          constraints=[problem.nonlinear_constraint([-np.inf, -np.inf], [np.inf, np.inf])])
    assert result.accepted_steps == problem.accepted_step == 1
    assert not np.array_equal(problem.accepted.parameters, before)
    assert result.fun <= problem.fun(before)
    with pytest.raises(ValueError, match="evaluate the candidate"):
        problem.accept_x(before + 0.123)
