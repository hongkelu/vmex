"""Public trial correction preserves prediction, gates and rejected evidence."""
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest

from vmex.core import freeboundary, freeboundary_continuation as fc
from vmex.core.freeboundary_problem import FreeBoundaryProblem
from vmex.core.projected_optimization import TrialRejected
from vmex.core.errors import VmecError


def callback(monkeypatch, tmp_path, *, converge=True, certify=True):
    import case
    from example_support import CaseRun
    from single_stage_support import FreeBoundaryRun

    anchor = NS(parameters=np.zeros(1), state=10., rcon0=1., zcon0=2.)
    result = NS(state=20., converged=converge, iterations=30,
                fsqr=1e-12, fsqz=1e-12, fsql=1e-12, fedge=1e-12)
    solve = Mock(return_value=NS(result=result, rcon0=11., zcon0=12.))
    tangent = Mock(return_value=4.)
    cert = Mock(side_effect=lambda cfg, point, state, **kw: NS(
        parameters=np.asarray(point), state=state, rcon0=11., zcon0=12.,
        root_residual_norm=1e-7, result=result))
    if not certify:
        cert.side_effect = VmecError('root residual gate')
    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', solve)
    monkeypatch.setattr(fc, 'certify_free_boundary_continuation_state', cert)
    monkeypatch.setattr(case, 'STATE_NAMES', ())
    monkeypatch.setattr(case, 'sampled_displacement', lambda _: np.zeros((1, 3)))
    problem = FreeBoundaryProblem.__new__(FreeBoundaryProblem)
    problem.x0 = problem.scales = np.ones(1)
    problem.deadline = None
    problem.accepted = anchor
    problem.cfg = NS(continuation_step=.1, max_continuation_steps=64)
    problem.solver = NS(resolution=None, edge_force_tolerance=1e-11,
                        implicit=NS(ftol=1e-11, max_iterations=12000))
    problem.inp = None
    problem.parameterization = lambda p: p
    problem._linearization = NS(tangent=tangent)
    problem._linearization_record = anchor
    problem._compact_jac = np.zeros((4, 1))
    problem._compact_state = lambda _: np.zeros(4)
    problem.metadata = {'holder': {'failed_trials': 0}}
    recorder = NS(problem=problem, args=NS(equilibrium_predictor='reused_dense'),
                  step=1801, output=tmp_path, accepted=anchor, values=np.zeros(4),
                  builder=NS(motion_bounds=lambda _: (0., 0.)), event=Mock())
    for name in ('_record_proposal', '_record_ordinary_correction', '_record_certification', '_record_candidate'):
        setattr(recorder, name, MethodType(getattr(FreeBoundaryRun, name), recorder))
    problem._emit = MethodType(CaseRun.solver_event, recorder)
    return problem, solve, tangent, cert


def test_tangent_path_retains_prediction(monkeypatch, tmp_path):
    problem, solve, tangent, _ = callback(monkeypatch, tmp_path)
    problem.evaluate_trial(np.array([.15]), 1)
    tangent.assert_called_once()
    assert [c.kwargs['initial_state'] for c in solve.call_args_list] == [12., 22.]


@pytest.mark.parametrize('converge,certify', [(False, True), (True, False)])
def test_reused_tangent_rejects_failed_equilibrium_and_certification(monkeypatch, tmp_path, converge, certify):
    problem, solve, tangent, cert = callback(monkeypatch, tmp_path, converge=converge, certify=certify)
    anchor = problem.accepted
    with pytest.raises(TrialRejected):
        problem.evaluate_trial(np.array([.05]), 1)
    tangent.assert_called_once()
    assert problem.accepted is anchor
    if not converge:
        cert.assert_not_called()
    with np.load(tmp_path/'trial_step_1802_trial_01_candidate.npz') as data:
        assert not bool(data['eligible_for_resume'])


def test_each_backtracking_trial_restarts_at_the_accepted_anchor(monkeypatch, tmp_path):
    problem, solve, _, _ = callback(monkeypatch, tmp_path)
    problem.evaluate_trial(np.array([.15]), 1)
    problem.evaluate_trial(np.array([.05]), 2)
    assert [c.kwargs['initial_state'] for c in solve.call_args_list] == [12., 22., 14.]
    assert [c.kwargs['constraint_continuation'] for c in solve.call_args_list] == [(1., 2.), (11., 12.), (1., 2.)]
    for c in solve.call_args_list:
        assert c.kwargs['ftol'] == c.kwargs['edge_force_tolerance'] == 1e-11
        assert c.kwargs['include_edge_in_convergence'] and c.kwargs['jacobian_retries'] == 0


def test_failed_derivative_preparation_never_launches_correction(monkeypatch, tmp_path):
    problem, solve, tangent, cert = callback(monkeypatch, tmp_path)
    problem._linearization_record = None
    monkeypatch.setattr(problem, '_derivatives', Mock(side_effect=RuntimeError('derivative gate failed')))
    with pytest.raises(RuntimeError, match='derivative gate failed'):
        problem.evaluate_trial(np.array([.05]), 1)
    solve.assert_not_called()
    tangent.assert_not_called()
    cert.assert_not_called()
