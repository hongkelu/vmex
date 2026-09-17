"""Exercise the actual trial callback without importing the GPU runtime."""
import ast
import time
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest

CASE = Path(__file__).resolve().parents[1]
ENTRY = CASE/'single_stage_problem.py'
RUNNERS = [pytest.param(ENTRY, id='maintained')]


class Rejected(Exception):
    pass


class CertificationError(Exception):
    pass


def callback(path, tmp_path, mode='reused_dense', *, converge=True, certify=True, prepared=True):
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_trial')
    fn.body = [n for n in fn.body if not isinstance(n, (ast.Nonlocal, ast.Import, ast.ImportFrom))]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    anchor = NS(parameters=np.zeros(1), state=10., dof_mask=None, rcon0=1., zcon0=2.)
    result = NS(state=20., converged=converge, iterations=30, fsqr=1e-12, fsqz=1e-12, fsql=1e-12, fedge=1e-12)
    solve = Mock(return_value=NS(result=result, rcon0=11., zcon0=12.))
    tangent = Mock(return_value=4.)
    certification = Mock(return_value=NS(state=20., rcon0=11., zcon0=12., root_residual_norm=1e-7, result=result))
    if not certify:
        certification.side_effect = CertificationError('root residual gate')
    env = dict(args=NS(equilibrium_predictor=mode), time=time, deadline=time.monotonic()+60,
               np=np, jnp=np, jax=NS(tree=NS(map=lambda f, x, dx: f(x, dx))),
               accepted=anchor, PARAMETER_SCALES=np.ones(1), step=1801, output=tmp_path,
               values=np.zeros(4), jac=np.zeros((4, 1)), motion_bounds=lambda _: (0., 0.),
               displacement=lambda _: np.zeros((1, 3)), event=Mock(),
               contract=dict(max_continuation_steps=64, force_tolerance=1e-11,
                             max_iterations=12000, edge_force_tolerance=1e-11),
               TrialRejected=Rejected, VmecError=CertificationError, STATE_NAMES=(),
               cfg=NS(params=None), solver=NS(resolution=None), inp=None, builder=lambda p:p,
               linearization=NS(tangent=tangent) if prepared else None, _solve_free_boundary_stage=solve,
               fc=NS(certify_free_boundary_continuation_state=certification), rows=lambda _:np.zeros(4))
    exec(compile(module, str(path), 'exec'), env)
    if fn.args.args[0].arg == 'self':
        obj = NS(**env, parameter_scales=env['PARAMETER_SCALES'])
        # Execute the real recorders too: rejected candidates must remain
        # diagnostic-only after moving the scientific callback into the script.
        support = ast.parse((CASE/'single_stage_support.py').read_text())
        for name in ('_record_proposal', '_record_ordinary_correction',
                     '_record_certification', '_record_candidate'):
            helper = next(n for n in ast.walk(support)
                          if isinstance(n, ast.FunctionDef) and n.name == name)
            helper.body = [n for n in helper.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
            compiled = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
            exec(compile(compiled, str(CASE/'single_stage_support.py'), 'exec'), env)
            setattr(obj, name, MethodType(env[name], obj))
        def call(delta, trial):
            return env['evaluate_trial'](obj, delta, trial)
    else:
        call = env['evaluate_trial']
    return call, solve, tangent, certification


@pytest.mark.parametrize('path', RUNNERS)
def test_tangent_path_retains_prediction(path, tmp_path):
    fn, solve, tangent, cert = callback(path, tmp_path, 'reused_dense')
    fn(np.array([.15]), 1)
    tangent.assert_called_once()
    assert [c.kwargs['initial_state'] for c in solve.call_args_list] == [12., 22.]


@pytest.mark.parametrize('path', RUNNERS)
@pytest.mark.parametrize('converge,certify', [(False, True), (True, False)])
def test_reused_tangent_rejects_failed_equilibrium_and_certification(path, tmp_path, converge, certify):
    fn, solve, tangent, cert = callback(path, tmp_path, 'reused_dense', converge=converge, certify=certify)
    with pytest.raises(Rejected):
        fn(np.array([.05]), 1)
    tangent.assert_called_once()
    if not converge:
        cert.assert_not_called()
    with np.load(tmp_path/'trial_step_1802_trial_01_candidate.npz') as data:
        assert not bool(data['eligible_for_resume'])


def test_each_backtracking_trial_restarts_at_the_accepted_anchor(tmp_path):
    fn, solve, tangent, cert = callback(ENTRY, tmp_path)
    fn(np.array([.15]), 1)
    fn(np.array([.05]), 2)
    assert [c.kwargs['initial_state'] for c in solve.call_args_list] == [12., 22., 14.]
    assert [c.kwargs['constraint_continuation'] for c in solve.call_args_list] == [(1.,2.),(11.,12.),(1.,2.)]
    for c in solve.call_args_list:
        assert c.kwargs['ftol'] == c.kwargs['edge_force_tolerance'] == 1e-11
        assert c.kwargs['include_edge_in_convergence'] and c.kwargs['jacobian_retries'] == 0


def test_missing_linearization_fails_instead_of_recomputing_slow_tangent(tmp_path):
    fn, solve, tangent, cert = callback(ENTRY, tmp_path, prepared=False)
    with pytest.raises(RuntimeError,match='current accepted-root dense linearization'):
        fn(np.array([.05]),1)
    solve.assert_not_called()
    tangent.assert_not_called()
    cert.assert_not_called()
