"""Coupled root refinement with real dense/FGMRES kernels on an analytic root."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex import optimize as opt
from vmex.core import freeboundary_problem as api
from vmex.core import _freeboundary_root_polish as polish


@dataclass(frozen=True, eq=False)
class Root:
    parameters: object
    state: object
    dof_mask: object
    _owner: object
    root_residual_norm: float
    rcon0: object = None
    zcon0: object = None
    result: object = None


@pytest.fixture
def coupled(monkeypatch):
    jax.config.update('jax_enable_x64', True)
    # Third coordinate is inactive and must remain exactly unchanged.
    mask = jnp.array([1., 1., 0.])
    matrix = jnp.array([[3., 1., 0.], [-1., 4., 0.], [0., 0., 0.]])
    coupling = jnp.array([[1., -2.], [3., 1.], [0., 0.]])
    params = jnp.array([.6, -.2, 0.])
    residual = jax.jit(lambda z, p, x, *_: matrix@z + .1*z*z*mask - p - coupling@x)

    class Chart:
        x0 = np.zeros(2)
        scales = np.ones(2)
        dof_names = ('a', 'b')

        def __call__(self, x):
            return x

        def coils_from_x(self, x):
            return x

    chart = Chart()
    inp = SimpleNamespace(lfreeb=True)
    solver = SimpleNamespace(implicit=SimpleNamespace(inp=inp, device=None, lconm1=False,
        ftol=1e-15, max_iterations=10), resolution=None, edge_force_tolerance=1e-15,
        adjoint_dense_batch_size=2, adjoint_dense_max_dofs=10, adjoint_solver='forward_dense_jax',
        adjoint_fail='error', adjoint_residual_rtol=1e-9, include_edge_in_convergence=True,
        field_from_parameters=chart)
    owner = object()
    initial = jnp.array([.2, -.01, 7.])
    anchor = Root(chart.x0.copy(), initial, mask, owner,
        float(jnp.linalg.norm(residual(initial*mask, params, chart.x0))),
        jnp.array([2.]), jnp.array([3.]), SimpleNamespace(iterations=5))
    cfg = SimpleNamespace(_owner=owner, _anchor=anchor, params=params, solver=solver,
        parameter_scales=chart.scales, continuation_step=.1, max_continuation_steps=64,
        root_residual_atol=1.)
    monkeypatch.setattr(polish.fbi, '_projected_residual', lambda *_: residual)
    monkeypatch.setattr(polish.im, '_dof_projector', lambda _, m: lambda x: x*m)
    monkeypatch.setattr(api.im, 'runtime_from_params', lambda *_: None)
    monkeypatch.setattr(polish, 'geometry_valid', lambda *a, **kw: True)

    def certify(config, point, state, *, rcon0, zcon0, **kw):
        assert config is cfg
        assert np.array_equal(rcon0, anchor.rcon0) and np.array_equal(zcon0, anchor.zcon0)
        assert float(state[2]) == 7.
        norm = float(jnp.linalg.norm(residual(state*mask, params, point)))
        # Fresh force diagnostics, not the original ordinary solve's result.
        # refine uses dataclasses.replace on a real result; supply its equivalent.
        @dataclass
        class Result:
            iterations: int = 0
            converged: bool = True
            fsqr: float = 1e-22
            fsqz: float = 1e-22
            fsql: float = 1e-22
            fedge: float = 1e-22
        result = Result()
        return Root(np.asarray(point).copy(), state, mask, owner, norm, rcon0, zcon0, result)

    monkeypatch.setattr(api.fc, 'certify_free_boundary_continuation_state', certify)
    problem = opt.FreeBoundaryProblem.from_loss(inp,
        lambda s, rt, c: .5*jnp.sum(s[:2]**2)+jnp.sum(c*c),
        quantities=(lambda s, rt: s[0], lambda s, rt: s[1]),
        parameterization=chart, continuation=cfg)
    yield problem, residual
    problem.close()


@pytest.mark.usefixtures('_module_jit_enabled')
def test_polished_seed_and_total_gradient_use_actual_kernels(coupled):
    p, residual = coupled
    old = p.accepted
    old_gradient = p.grad(p.x0).copy()
    p.enable_root_polishing()
    assert p.accepted is not old and p.accepted_step == 0
    assert p.accepted.root_residual_norm <= 1e-12
    assert p.accepted.state[2] == 7.
    assert p.accepted.result.iterations == 5
    p.enable_matrix_free(np.array([.001, -.002]), refresh_horizon=10)
    z = p.accepted.state
    matrix = jax.jacfwd(residual, 0)(z*jnp.array([1., 1., 0.]), p.params, jnp.zeros(2))[:2,:2]
    coupling = jax.jacfwd(residual, 2)(z*jnp.array([1., 1., 0.]), p.params, jnp.zeros(2))[:2]
    expected = np.asarray(z[:2]) @ -np.linalg.solve(matrix, coupling)
    np.testing.assert_allclose(p.grad(p.x0), expected, rtol=1e-10, atol=1e-12)
    assert not np.allclose(old_gradient, expected)


@pytest.mark.usefixtures('_module_jit_enabled')
def test_polish_failure_keeps_initial_root_and_caches(coupled):
    p, _ = coupled
    original = p.accepted
    grad = p.grad(p.x0).copy()
    with pytest.raises(polish.RootPolishError, match='budget exhausted'):
        p.enable_root_polishing(tolerance=1e-14, max_steps=1)
    assert p.accepted is original and p._root_polish_options is None
    np.testing.assert_array_equal(p.grad(p.x0), grad)


@pytest.mark.usefixtures('_module_jit_enabled')
def test_polished_coil_quantity_matches_independent_endpoints(coupled, monkeypatch):
    """A moving-surface-like row retains both direct and root-response terms."""
    from vmex.core import freeboundary
    p, _ = coupled
    q = opt.FreeBoundaryProblem.from_loss(p.inp, lambda s, rt, c: jnp.sum(s[:2]**2),
        coil_quantities=(lambda s, rt, c: s[0]*c[0] + s[1] + 2*c[1],),
        parameterization=p.parameterization, continuation=p.cfg)
    try:
        q.enable_root_polishing()
        anchor = q.accepted

        def ordinary(inp, *, initial_state, **kwargs):
            return SimpleNamespace(result=SimpleNamespace(state=initial_state, converged=True, iterations=3),
                                   rcon0=anchor.rcon0, zcon0=anchor.zcon0)

        monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', ordinary)
        direction = np.array([.2, -.3])
        expected = q.constraint_jac(q.x0) @ direction
        for h in (3e-4, 1e-4):
            plus, vp = q.evaluate_trial(h*direction, predict=False)
            minus, vm = q.evaluate_trial(-h*direction, predict=False)
            assert max(plus.root_residual_norm, minus.root_residual_norm) <= 1e-12
            np.testing.assert_allclose((vp[1:]-vm[1:])/(2*h), expected, rtol=1e-7, atol=1e-9)
            assert q.accepted is anchor and q.accepted_step == 0
    finally:
        q.close()


@pytest.mark.usefixtures('_module_jit_enabled')
def test_trial_polishing_preserves_accepted_seed_and_recertifies(coupled, monkeypatch):
    from vmex.core import freeboundary
    p, residual = coupled
    p.enable_root_polishing()
    p.enable_matrix_free(refresh_horizon=10)
    anchor, seed = p.accepted, p._preconditioner
    calls = []

    def ordinary(inp, *, external_field, initial_state, **kwargs):
        calls.append(np.asarray(initial_state).copy())
        # An imperfect ordinary root: Newton polishing must finish the solve.
        return SimpleNamespace(result=SimpleNamespace(state=initial_state, converged=True, iterations=3),
                               rcon0=anchor.rcon0, zcon0=anchor.zcon0)

    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', ordinary)
    point = np.array([.001, -.002])
    trial, rows = p.evaluate_trial(point, predict=False)
    assert trial.root_residual_norm < 1e-12 and np.all(np.isfinite(rows))
    np.testing.assert_array_equal(calls[-1], anchor.state)
    assert p.accepted is anchor and p._preconditioner is seed
    p.accept(trial)
    assert p.accepted is trial and p.accepted_step == 1


@pytest.mark.usefixtures('_module_jit_enabled')
def test_polished_fd_trial_must_keep_requested_force_tolerance(coupled, monkeypatch):
    from vmex.core import freeboundary
    p, _ = coupled
    p.enable_root_polishing()
    p.enable_matrix_free()
    anchor = p.accepted
    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', lambda inp, **kw:
        SimpleNamespace(result=SimpleNamespace(state=kw['initial_state'], converged=True),
                        rcon0=anchor.rcon0, zcon0=anchor.zcon0))
    with pytest.raises(api.TrialRejected, match='requested force tolerance'):
        p.evaluate_trial(np.array([.001, -.002]), predict=False, ftol=1e-24)
    assert p.accepted is anchor


@pytest.mark.parametrize('tolerance,steps', [(0.,3),(float('nan'),3),(1e-12,0),(1e-12,True)])
def test_invalid_polishing_contract(coupled, tolerance, steps):
    p, _ = coupled
    with pytest.raises(ValueError, match='positive finite'):
        p.enable_root_polishing(tolerance=tolerance, max_steps=steps)


def test_one_dense_retry_releases_temporary_factors(monkeypatch):
    calls = []
    class Temporary:
        def close(self):
            calls.append('close')
    def refine(record, cfg, seed, **kwargs):
        calls.append('refine')
        if seed == 'old':
            raise polish.RootPolishError('stale LU')
        return 'polished', {'seconds': .1}
    monkeypatch.setattr(polish, 'refine', refine)
    result = polish.polish_with_recovery('root','config','old',
        lambda root:(Temporary(),Temporary()), lambda event:None)
    assert result == 'polished' and calls == ['refine','refine','close','close']


def test_failed_dense_retry_is_bounded_and_releases_factors(monkeypatch):
    calls = []
    class Temporary:
        def close(self):
            calls.append('close')
    def fail(*args, **kwargs):
        calls.append('refine')
        raise polish.RootPolishError('root did not improve')
    monkeypatch.setattr(polish, 'refine', fail)
    with pytest.raises(polish.RootPolishError, match='did not improve'):
        polish.polish_with_recovery('root','config','old',
            lambda root:(Temporary(),Temporary()), lambda event:None)
    assert calls == ['refine','refine','close','close']


@pytest.mark.usefixtures('_module_jit_enabled')
def test_already_polished_initial_state_is_not_changed(coupled):
    p, _ = coupled
    p.enable_root_polishing()
    anchor = p.accepted
    p.enable_root_polishing()
    assert p.accepted is anchor and p.accepted_step == 0


@pytest.mark.usefixtures('_module_jit_enabled')
def test_host_inactive_drift_is_removed_even_below_root_tolerance(coupled):
    p, _ = coupled
    p.enable_root_polishing()
    p.enable_matrix_free()
    anchor = p.accepted
    drifted = replace(anchor, state=anchor.state.at[2].add(1e-5))
    repaired = p._polish_record(drifted)
    assert repaired.state[2] == anchor.state[2] == 7.
    assert repaired.root_residual_norm <= 1e-12
    assert p.accepted is anchor


def test_invalid_newton_geometry_is_rejected_without_promotion(coupled, monkeypatch):
    p, _ = coupled
    anchor = p.accepted
    monkeypatch.setattr(polish, 'geometry_valid', lambda *a, **kw: False)
    with pytest.raises(polish.RootPolishError, match='starting geometry'):
        p.enable_root_polishing()
    assert p.accepted is anchor and p._root_polish_options is None


@pytest.mark.parametrize('field', ['rcon0', 'zcon0', 'dof_mask'])
def test_polished_fast_path_rejects_changed_anchor_contract(coupled, field):
    p, _ = coupled
    p.enable_root_polishing()
    altered = replace(p.accepted, **{field: getattr(p.accepted, field)+1})
    with pytest.raises(polish.RootPolishError, match=field):
        p._polish_record(altered)


@pytest.mark.usefixtures('_module_jit_enabled')
@pytest.mark.parametrize('iteration,predict,expected_calls', [(1, True, 2), (2, True, 1), (1, False, 1)])
def test_initial_bad_predictor_retries_unchanged_anchor_once(coupled, monkeypatch, iteration, predict, expected_calls):
    from vmex.core import freeboundary
    from vmex.core.errors import VmecJacobianError
    p, _ = coupled
    p.enable_root_polishing()
    p.enable_matrix_free()
    anchor, calls = p.accepted, []

    def ordinary(inp, *, initial_state, constraint_continuation, **kw):
        calls.append(np.asarray(initial_state).copy())
        np.testing.assert_array_equal(constraint_continuation[0], anchor.rcon0)
        if len(calls) == 1:
            raise VmecJacobianError('bad predictor', iteration=iteration)
        np.testing.assert_array_equal(initial_state, anchor.state)
        return SimpleNamespace(result=SimpleNamespace(state=initial_state, converged=True, iterations=3),
                               rcon0=anchor.rcon0, zcon0=anchor.zcon0)

    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', ordinary)
    if expected_calls == 2:
        root, _ = p.evaluate_trial(np.array([.001, -.002]), predict=predict)
        assert root.root_residual_norm <= 1e-12
    else:
        with pytest.raises(api.TrialRejected):
            p.evaluate_trial(np.array([.001, -.002]), predict=predict)
    assert len(calls) == expected_calls and p.accepted is anchor
