"""Exercise the readable loop with real acceptance rules and synthetic rows."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from vmex.core import projected_optimization as optimizer
from vmex.core.coil_parameters import CoilParameters

import numpy as np
import pytest


@pytest.fixture
def problem_and_loop(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))

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






def test_example_passes_its_definitions_to_the_problem(problem_and_loop, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from vmex import optimize as opt

    example = importlib.import_module("single_stage_free_boundary_optimization")
    calls = []
    problem = SimpleNamespace(dof_names=("current[1]/nominal",))

    class Run:
        def __init__(self, args):
            self.cfg = SimpleNamespace(solver=SimpleNamespace(implicit=SimpleNamespace(inp="input")))
            self.builder, self.resume, self.deadline = "chart", None, 100.
            self.input, self.coil_parameters, self.continuation = "input", "chart", self.cfg
            self.output = tmp_path

        def __enter__(self):
            return self

        def bind(self, value):
            assert value is problem

        def __exit__(self, *exc):
            calls.append(self.status)

    def build(inp, terms, **kwargs):
        calls.append((inp, terms, kwargs))
        return problem

    monkeypatch.setattr(example, "parse_args", lambda *a, **k: SimpleNamespace(
        output_dir=tmp_path, target_step=10, initialize_only=True))
    monkeypatch.setattr(example, "CaseRun", Run)
    monkeypatch.setattr(opt.FreeBoundaryProblem, "from_tuples", build)
    monkeypatch.setattr(opt, "OptimizationMonitor", lambda p: None)
    example.main([])
    inp, terms, settings = calls[0]
    assert inp == "input" and settings['parameterization'] == "chart"
    assert terms[0][1:] == (0.0, 1.0)
    assert [c.target for c in settings['constraints']] == [example.IOTA_TARGET, example.ASPECT_TARGET, example.B0_TARGET]
    assert [c.rtol for c in settings['constraints']] == [.01]*3
    assert settings['objective_normalization'] == 'initial'
    assert calls[1] == 'initialized_only'
