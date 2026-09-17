"""Public API, derivatives, trial ownership and bounds on an analytic root.

These tests exercise the real API/optimizer with a known implicit equilibrium;
they do not claim qualification of the 48x40 plasma equilibrium.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex import optimize as opt
from vmex.core import freeboundary_problem as api
from vmex.core.coil_parameters import CoilParameters
from vmex.core.projected_optimization import TrialRejected


@dataclass(frozen=True, eq=False)
class Root:
    parameters: np.ndarray
    state: object
    rcon0: object = None
    zcon0: object = None
    dof_mask: object = None
    root_residual_norm: float = 0.0
    result: object = None


@dataclass(frozen=True, eq=False)
class Config:
    solver: object
    params: object
    _anchor: Root
    parameter_scales: object
    continuation_step: float = 0.1
    max_continuation_steps: int = 64


@pytest.fixture
def analytic(monkeypatch):
    jax.config.update("jax_enable_x64", True)
    size = 5
    matrix = np.eye(size) + 0.03 * np.ones((size, size))
    base = np.array([0.2, 5.0, 0.7, -0.3, 0.8])

    def state(x):
        return jnp.asarray(base) + jnp.asarray(matrix) @ x + 0.1 * x * x

    def derivative(x):
        return matrix + 0.2 * np.diag(np.asarray(x))

    class Chart:
        x0 = np.zeros(size)
        scales = np.ones(size)
        dof_names = tuple(f"coil[{i}]" for i in range(size))

        def __call__(self, x):
            return x

        def motion_bounds(self, delta):
            return float(np.max(np.abs(delta))), 0.0

        def coils_from_x(self, x):
            return np.asarray(x).copy()

    chart = Chart()
    inp = SimpleNamespace(lfreeb=True)
    solver = SimpleNamespace(
        implicit=SimpleNamespace(inp=inp, ftol=1e-11, max_iterations=10),
        resolution=None,
        edge_force_tolerance=1e-11,
        field_from_parameters=chart,
        include_edge_in_convergence=True,
        adjoint_solver="forward_dense_jax",
    )
    anchor = Root(chart.x0.copy(), state(chart.x0))
    cfg = Config(solver, None, anchor, chart.scales)
    stats = dict(rhs=[], solves=0, closed=0, fail=False, gradient_calls=0)
    monkeypatch.setattr(api.im, "runtime_from_params", lambda *a: None)

    def pullback(record, config, rhs, *, diagnostics=None, return_linearization=False):
        stats["rhs"].append(rhs.shape[0])
        jac = np.asarray(rhs) @ derivative(record.parameters)
        if not return_linearization:
            return jac

        class Linearization:
            field_jacobian = jac

            def tangent(self, current, current_config, delta, **kwargs):
                assert current is record and current_config is config
                return jnp.asarray(derivative(record.parameters) @ delta)

            def close(self):
                stats["closed"] += 1

        return Linearization()

    def correct(inp, *, external_field, **kwargs):
        stats["solves"] += 1
        return SimpleNamespace(
            result=SimpleNamespace(state=state(external_field), converged=not stats["fail"]), rcon0=None, zcon0=None
        )

    def certify(config, point, value, **kwargs):
        np.testing.assert_allclose(value, state(point), rtol=0, atol=0)
        return Root(np.asarray(point).copy(), value)

    monkeypatch.setattr(api.fc, "free_boundary_continuation_state_pullback", pullback)
    monkeypatch.setattr(api.fc, "certify_free_boundary_continuation_state", certify)
    monkeypatch.setattr(
        api.fc, "reanchor_free_boundary_continuation_config", lambda cfg, root: replace(cfg, _anchor=root)
    )
    from vmex.core import freeboundary

    monkeypatch.setattr(freeboundary, "_solve_free_boundary_stage", correct)
    terms = [(lambda s, rt: s, 0.0, 2.0)]
    constraints = [
        opt.TargetBand(lambda s, rt: s[0], 0.2, scale=0.005),
        opt.TargetBand(lambda s, rt: s[1], 5.0, scale=0.05),
    ]
    problem = opt.FreeBoundaryProblem.from_tuples(
        inp, terms, parameterization=chart, continuation=cfg, constraints=constraints, objective_normalization=2.0
    )
    yield problem, stats, state, derivative
    problem.close()


def test_scalar_and_constraint_derivatives_share_compact_pullback(analytic):
    p, stats, state, derivative = analytic
    value, gradient = p.value_and_grad(p.x0)
    expected = np.asarray(state(p.x0))
    np.testing.assert_allclose(value, 0.25 * np.dot(expected, expected))
    np.testing.assert_allclose(gradient, 0.5 * expected @ derivative(p.x0), rtol=1e-13)
    np.testing.assert_allclose(p.constraint_jac(p.x0), derivative(p.x0)[:2], rtol=1e-13)
    p.value_and_grad(p.x0)
    assert stats["rhs"] == [3]  # Objective norm plus two constraints, not five residual rows.
    assert stats["solves"] == 0
    residual, jac = p.residual_and_jac(p.x0)
    np.testing.assert_allclose(jac.T @ residual, gradient, rtol=1e-13)
    assert stats["rhs"] == [3, 5]


def test_finite_differences_and_nonpromoting_evaluation(analytic):
    p, stats, _, _ = analytic
    anchor = p.accepted
    gradient = p.grad(p.x0)
    eps = 1e-5
    fd = []
    for index in range(p.x0.size):
        delta = np.eye(p.x0.size)[index] * eps
        fd.append((p.fun(p.x0 + delta) - p.fun(p.x0 - delta)) / (2 * eps))
    np.testing.assert_allclose(gradient, fd, rtol=2e-8, atol=1e-9)
    assert p.accepted is anchor
    candidate, _ = p.evaluate_trial(np.ones(5) * 0.001, 1)
    assert p.accepted is anchor
    p.accept(candidate)
    assert p.accepted is candidate and stats["closed"] > 0
    with pytest.raises(ValueError, match="current evaluation"):
        p.accept(anchor)


def test_failed_trial_keeps_anchor_and_reports_rejection(analytic):
    p, stats, _, _ = analytic
    before = p.accepted
    stats["fail"] = True
    with pytest.raises(TrialRejected, match="did not converge"):
        p.evaluate_trial(np.ones(5) * 0.001, 1)
    assert p.accepted is before
    result = opt.minimize_projected(p, maxiter=2)
    assert result.status == "stagnated" and result.nit == 0 and not result.success
    assert p.accepted is before


def test_optimizer_promotes_only_accepted_steps_and_preserves_reference(analytic):
    p, stats, _, _ = analytic
    events = []
    result = opt.minimize_projected(p, maxiter=2, initial_gradient_norm=2.0, callback=events.append)
    assert result.nit == 2 and result.status == "step_budget_reached" and not result.success
    assert result.initial_gradient_norm == 2.0
    assert len(events) == 2 and result.accepted is p.accepted
    assert result.fun < p.fun(p.x0)
    assert np.all(np.abs(result.constraints - p.targets) <= p.constraint_tolerances)


def test_returned_arrays_do_not_mutate_cache(analytic):
    p, *_ = analytic
    _, grad = p.value_and_grad(p.x0)
    expected = grad.copy()
    grad[:] = 999
    np.testing.assert_array_equal(p.grad(p.x0), expected)


def test_invalid_chart_owner_and_bands(analytic):
    p, *_ = analytic
    with pytest.raises(ValueError, match="identical field chart"):
        opt.FreeBoundaryProblem.from_tuples(
            p.inp, [(lambda s, rt: s, 0, 1)], parameterization=object(), continuation=p.cfg
        )
    with pytest.raises(ValueError):
        opt.TargetBand(lambda s, rt: s[0], 0.0, rtol=0.01)
    assert opt.TargetBand(lambda s, rt: s[0], 0.0, atol=0.2).tolerance == 0.2
    with pytest.raises(ValueError):
        opt.minimize_projected(p, maxiter=-1)


def test_general_coil_chart_and_continuous_motion_bound():
    rng = np.random.default_rng(38)
    shape = (2, 3, 7)
    coefficients = rng.normal(size=shape)
    currents = np.array([3e5, -2e5])
    chart = CoilParameters(coefficients, currents, current_dofs=(1,), max_coil_mode=2, nfp=3)
    x = rng.normal(size=chart.size) * 0.001
    assert chart.size == 31 and len(chart.dof_names) == 31
    np.testing.assert_allclose(chart.base_currents_at(x), [currents[0], currents[1] * (1 + x[0])])
    after = np.asarray(chart.curve_dofs_at(x))
    np.testing.assert_array_equal(after[:, :, 5:], coefficients[:, :, 5:])
    np.testing.assert_allclose(after[:, :, :5], coefficients[:, :, :5] + x[1:].reshape(2, 3, 5))
    t = np.arange(10001) / 10001
    basis = np.array(
        [np.ones_like(t), np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), np.sin(4 * np.pi * t), np.cos(4 * np.pi * t)]
    )
    movement = np.einsum("cdk,ks->csd", x[1:].reshape(2, 3, 5), basis)
    bound, current = chart.motion_bounds(x)
    assert np.max(np.linalg.norm(movement, axis=-1)) <= bound * (1 + 1e-12)
    assert current == abs(x[0])
    coefficients[:] = 0
    currents[:] = 0
    assert np.any(chart.coefficients) and np.any(chart.currents)


def test_zero_objective_has_finite_zero_gradient_and_converges(analytic):
    p, stats, _, _ = analytic
    zero = opt.FreeBoundaryProblem.from_tuples(
        p.inp,
        [(lambda state, rt: jnp.zeros(3), 0.0, 1.0)],
        parameterization=p.parameterization,
        continuation=p.cfg,
        constraints=p.constraints,
    )
    try:
        value, grad = zero.value_and_grad(zero.x0)
        assert value == 0.0 and np.all(grad == 0.0)
        result = opt.minimize_projected(zero, maxiter=1)
        assert result.success and result.nit == 0 and stats["solves"] == 0
    finally:
        zero.close()


def test_callback_stop_preserves_accepted_result(analytic):
    p, *_ = analytic

    def stop(result):
        raise StopIteration

    result = opt.minimize_projected(p, maxiter=10, callback=stop)
    assert result.status == "callback_stopped" and result.nit == 1
    assert result.accepted is p.accepted


def test_continuation_scale_and_edge_gates(analytic):
    p, *_ = analytic
    terms = [(lambda state, rt: state, 0.0, 1.0)]
    with pytest.raises(ValueError, match="scales differ"):
        opt.FreeBoundaryProblem.from_tuples(
            p.inp,
            terms,
            parameterization=p.parameterization,
            continuation=replace(p.cfg, parameter_scales=np.full(5, 2.0)),
        )
    p.solver.include_edge_in_convergence = False
    with pytest.raises(ValueError, match="edge-certified"):
        opt.FreeBoundaryProblem.from_tuples(p.inp, terms, parameterization=p.parameterization, continuation=p.cfg)


@pytest.mark.parametrize("failure", [None, "solve", "edge"])
def test_standalone_constructor_solves_once_and_checks_initial_root(analytic, monkeypatch, failure):
    p, stats, state, _ = analytic
    from vmex.core import freeboundary

    seen = {}

    def make_solver(inp, field, **options):
        seen["options"] = options
        return p.solver

    def solve(inp, *, external_field, **kwargs):
        stats["solves"] += 1
        seen["solve"] = kwargs
        return SimpleNamespace(
            result=SimpleNamespace(converged=failure != "solve", state=state(external_field), fedge=1e-12),
            rcon0=None,
            zcon0=None,
        )

    def certify(solver, params, point, **kwargs):
        seen["certify"] = kwargs
        anchor = Root(
            np.asarray(point), kwargs["state"], result=SimpleNamespace(fedge=1e-7 if failure == "edge" else 1e-12)
        )
        return replace(p.cfg, _anchor=anchor)

    monkeypatch.setattr(api.fbi, "make_free_boundary_config", make_solver)
    monkeypatch.setattr(api.im, "params_from_input", lambda inp: None)
    monkeypatch.setattr(freeboundary, "_solve_free_boundary_stage", solve)
    monkeypatch.setattr(api.fc, "make_free_boundary_continuation_config_from_state", certify)
    kwargs = dict(
        parameterization=p.parameterization,
        restart_from=p.accepted.state,
        constraints=p.constraints,
        solver_options=dict(ftol=1e-11),
    )
    if failure:
        with pytest.raises(api.VmecError):
            opt.FreeBoundaryProblem.from_tuples(p.inp, [(lambda s, rt: s, 0.0, 1.0)], **kwargs)
    else:
        other = opt.FreeBoundaryProblem.from_tuples(p.inp, [(lambda s, rt: s, 0.0, 1.0)], **kwargs)
        other.close()
        assert seen["certify"]["root_residual_atol"] == 2e-6
        assert seen["options"]["adjoint_solver"] == "forward_dense_jax"
    assert stats["solves"] == 1
    assert seen["solve"]["initial_state"] is p.accepted.state
    assert seen["solve"]["jacobian_retries"] == 0
    assert seen["solve"]["include_edge_in_convergence"]


def test_essos_coil_parameter_roundtrip():
    pytest.importorskip("essos.coils")
    rng = np.random.default_rng(483)
    chart = CoilParameters(
        rng.normal(size=(2, 3, 5)), [2e5, -3e5], current_dofs=(1,), nfp=2, stellsym=True, n_segments=24
    )
    coils = chart.coils_from_x(chart.x0)
    reconstructed = CoilParameters.from_coils(coils, current_dofs=(1,))
    np.testing.assert_array_equal(reconstructed.coefficients, chart.coefficients)
    np.testing.assert_array_equal(reconstructed.currents, chart.currents)
    assert reconstructed.nfp == 2 and reconstructed.stellsym and reconstructed.n_segments == 24


def test_complex_parameters_are_rejected_without_silent_conversion(analytic):
    problem, *_ = analytic
    with pytest.raises(ValueError, match="real"):
        problem.fun(problem.x0.astype(complex) + 1j)
    with pytest.raises(ValueError, match="real"):
        CoilParameters(np.zeros((1, 3, 3), dtype=complex), [1.0], current_dofs=(0,))


def test_equilibrium_preserves_standard_type_and_lazy_fixed_geometry_wout(analytic, monkeypatch):
    problem, stats, *_ = analytic
    seen = []
    output = object()

    def wout(record):
        seen.append(record)
        return output

    monkeypatch.setattr(problem, "_wout", wout)
    eq = problem.equilibrium_from_x(problem.x0)
    assert isinstance(eq, opt.Equilibrium)
    assert eq.solution is problem.accepted.state and seen == []
    assert eq.wout is output and eq.wout is output
    assert seen == [problem.accepted] and stats["solves"] == 0
