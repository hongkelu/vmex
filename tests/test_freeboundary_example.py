"""Exercise the readable loop with real acceptance rules and synthetic rows."""

from pathlib import Path
from types import SimpleNamespace
from vmex.core import projected_optimization as optimizer
from vmex.core.coil_parameters import CoilParameters

import numpy as np
import pytest


@pytest.fixture
def problem_and_loop(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples/optimization"))

    class Problem:
        def __init__(self):
            # No solver or output directory: only test iteration/acceptance.
            self.step = 8
            self.values = np.array([1., 0., 0., 0.])
            self.parameter_scales = np.ones(111)
            self.constraint_scales = np.array([.005, .05, .01])
            self.targets = np.array([.2, 5., -.17506474574437714])
            self.policy = optimizer.ProjectedOptions()
            self.initial_gradient_norm = 2.0  # Simulate a resumed run.
            self.jac = np.zeros((4, 111))
            self.jac[0, 6] = 1.
            self.jac[1, 3] = self.jac[2, 4] = self.jac[3, 5] = 1.
            self.events = []
            self.trials = 0
            self.promotions = []
            self.failure = None
            self.scales = self.parameter_scales
            self.parameterization = CoilParameters(np.zeros((4, 3, 9)), np.ones(4), current_dofs=(1, 2, 3))
            self.constraint_tolerances = 0.01 * np.abs(self.targets)
            self.accepted = SimpleNamespace(parameters=np.zeros(111), values=self.values)

        def event(self, phase, **values):
            self.events.append((phase, values))

        def linearize(self):
            return self.values, self.jac

        def evaluate_trial(self, delta, trial):
            self.trials += 1
            if self.failure is not None:
                raise self.failure
            values = self.values + self.jac @ delta
            return SimpleNamespace(parameters=self.accepted.parameters + delta, values=values), values

        def accept(self, trial):
            self.promotions.append(trial)
            self.accepted = trial
            self.values = trial.values
            self.step += 1

        def optimizer_rows(self, record):
            return record.values

        def constraint_values(self, x):
            return self.targets + self.values[1:] * self.constraint_scales

    def optimize(problem, target_step):
        result = optimizer.minimize_projected(
            problem, maxiter=target_step - problem.step, options=problem.policy,
            initial_gradient_norm=problem.initial_gradient_norm)
        assert result.initial_gradient_norm == problem.initial_gradient_norm
        return result.status

    return Problem(), optimize, optimizer.TrialRejected


def test_absolute_budget_and_original_gradient_reference(problem_and_loop):
    problem, optimize, _ = problem_and_loop
    assert optimize(problem, 10) == "step_budget_reached"
    assert problem.step == 10 and len(problem.promotions) == 2
    assert problem.values[0] < 1.
    assert problem.initial_gradient_norm == 2.0


def test_failed_equilibria_exhaust_trials_without_promotion(problem_and_loop):
    problem, optimize, rejected = problem_and_loop
    problem.failure = rejected("equilibrium failed certification")
    before = problem.values.copy()
    assert optimize(problem, 10) == "stagnated"
    assert problem.trials == 6 and problem.step == 8
    assert problem.promotions == []
    np.testing.assert_array_equal(problem.values, before)


def test_stationarity_stops_before_trial(problem_and_loop):
    problem, optimize, _ = problem_and_loop
    problem.jac[0] = 0.
    assert optimize(problem, 10) == "converged"
    assert problem.trials == 0 and problem.step == 8


def test_programming_error_propagates_without_promotion(problem_and_loop):
    problem, optimize, _ = problem_and_loop
    problem.failure = ValueError("unexpected implementation error")
    with pytest.raises(ValueError, match="unexpected implementation error"):
        optimize(problem, 10)
    assert problem.trials == 1 and problem.promotions == []


def test_already_at_budget_does_not_evaluate(problem_and_loop):
    problem, optimize, _ = problem_and_loop
    assert optimize(problem, 8) == "step_budget_reached"
    assert problem.trials == 0 and problem.promotions == []







def test_standalone_script_passes_original_physics_and_policy(monkeypatch, tmp_path):
    import runpy
    from unittest.mock import Mock
    from dataclasses import asdict
    import json
    pytest.importorskip('essos')
    from vmex import optimize as opt
    folder = Path(__file__).resolve().parents[1] / 'examples/optimization'
    monkeypatch.syspath_prepend(str(folder))
    entry = runpy.run_path(str(folder/'single_stage_free_boundary_optimization.py'))
    calls = []
    class Output:
        deadline = None
        solver_event = Mock()
        def __init__(self, *a, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            calls.append(self.status)
        def bind(self, problem, **kw):
            assert problem.accepted_step == 0
    def create(inp, terms, **kw):
        calls.append((inp, terms, kw))
        return SimpleNamespace(accepted_step=0)
    monkeypatch.setattr(opt.FreeBoundaryProblem, 'from_tuples', create)
    entry['main'].__globals__['Diagnostics'] = Output
    entry['main'](['--output-dir',str(tmp_path/'run'),'--device','cpu','--initialize-only'])
    inp, terms, settings = calls[0]
    assert (inp.ntheta,inp.nzeta,list(inp.ns_array)) == (48,40,[31])
    assert settings['parameterization'].size == 111
    assert settings['parameterization'].n_segments == 75
    assert settings['solver_options']['ftol'] == settings['solver_options']['edge_force_tolerance'] == 1e-11
    assert settings['solver_options']['adjoint_dense_batch_size'] == 32
    assert [c.target for c in settings['constraints']] == [.2,5.,-.17506474574437714]
    expected = json.loads((Path(__file__).parent/'data/free_boundary_qa/optimizer_policy.json').read_text())
    assert settings['checkpoint_identity']['optimizer'] == expected == asdict(opt.ProjectedOptions())
    assert settings['restart_from'].name == 'initial_state.npz'
    assert calls[1] == 'initialized_only'
