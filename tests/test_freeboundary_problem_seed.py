"""FreeBoundaryProblem.from_loss seeds a cold deck as main's implicit free boundary does (uwplasma/vmex#451 review).

The review's deck: LP QA with its ESSOS coils, ns 15, mpol = ntor = 3, ftol 1e-10. A cold seed (no ``restart_from``)
used to forbid the axis re-guess and the Jacobian retries of an ordinary solve, so with ``RAXIS_CC = 0`` it failed
the fused-vacuum axis check and with the deck's axis it stopped on "initial Jacobian changed sign", while main's
``solve_free_boundary_implicit`` solves the same deck cold.  Both seeds must now solve, and the accepted-state
aspect ratio and its coil gradient must agree with main's anchored implicit gradient.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.full

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
NS, MODES, FTOL = 15, 3, 1e-10


@pytest.fixture(scope="module")
def case():
    return _case()


def _case():
    """The review deck, its ESSOS field map and the zero design point."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    essos = pytest.importorskip("essos")
    import jax.numpy as jnp
    from essos.coils import Coils
    from essos.fields import BiotSavart

    import vmex as vj

    del essos
    coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))
    base = coils.dofs
    n_curves = coils.dofs_curves.size
    scales = jnp.concatenate((jnp.full(n_curves, 0.02), 0.20 * jnp.abs(base[n_curves:])))

    def field_from_parameters(x):
        return BiotSavart(coils.with_dofs(base + scales * x))

    inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres").change_resolution(
        mpol=MODES, ntor=MODES, ntheta=2 * MODES + 6, nzeta=16)
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field", phiedge=-0.025, ns_array=np.array([NS]),
                  niter_array=np.array([20000]), ftol_array=np.array([FTOL]))
    return inp, field_from_parameters, np.zeros(base.size)


def _main_gradient(inp, field_from_parameters, x0):
    import jax
    import jax.numpy as jnp

    from vmex.core import implicit as im
    from vmex.core.freeboundary_implicit import make_free_boundary_config, solve_free_boundary_implicit

    params = im.params_from_input(inp)
    config = make_free_boundary_config(inp, field_from_parameters(jnp.asarray(x0)), ns=NS, ftol=FTOL,
                                       max_iterations=20000, adjoint_tol=1e-9,
                                       field_from_parameters=field_from_parameters)
    context = im.runtime_from_params(params, config.implicit)
    value, gradient = jax.value_and_grad(
        lambda x: im.aspect_ratio(solve_free_boundary_implicit(params, x, config), context))(jnp.asarray(x0))
    return float(value), np.asarray(gradient)


@pytest.mark.parametrize("axis", ["deck", "zero"])
def test_from_loss_seeds_cold_like_main(case, axis):
    from vmex.core import implicit as im
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp, field_from_parameters, x0 = case
    if axis == "zero":
        inp = replace(inp, raxis_c=np.zeros_like(inp.raxis_c), zaxis_s=np.zeros_like(inp.zaxis_s))
    reference, reference_gradient = _main_gradient(inp, field_from_parameters, x0)
    problem = FreeBoundaryProblem.from_loss(inp, lambda state, runtime, x: im.aspect_ratio(state, runtime), x0,
                                            field_from_parameters=field_from_parameters)
    try:
        value, gradient = problem.value_and_grad(x0)
    finally:
        problem.close()
    relative = np.linalg.norm(gradient - reference_gradient) / np.linalg.norm(reference_gradient)
    print(f"axis={axis}: aspect {value:.9f} vs main {reference:.9f}, gradient relative difference {relative:.2e}")
    assert abs(value - reference) <= 1e-5 * abs(reference)
    assert relative <= 1e-3
