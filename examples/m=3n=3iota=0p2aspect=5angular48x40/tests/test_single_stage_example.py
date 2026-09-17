"""Exercise the readable loop with real acceptance rules and synthetic rows."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def problem_and_loop(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    example = importlib.import_module("single_stage_free_boundary_optimization")
    support = importlib.import_module("single_stage_support")
    optimizer = importlib.import_module("optimization")

    class Problem(support.FreeBoundaryRun):
        def __init__(self):
            # No solver or output directory: only test iteration/acceptance.
            self.step = 8
            self.values = np.array([1., 0., 0., 0.])
            self.parameter_scales = np.ones(111)
            self.constraint_scales = np.array([.005, .05, .01])
            self.targets = np.array([.2, 5., -.17506474574437714])
            self.policy = optimizer.Policy()
            self.initial_gradient_norm = 2.0  # Simulate a resumed run.
            self.jac = np.zeros((4, 111))
            self.jac[0, 6] = 1.
            self.jac[1, 3] = self.jac[2, 4] = self.jac[3, 5] = 1.
            self.events = []
            self.trials = 0
            self.promotions = []
            self.failure = None

        def event(self, phase, **values):
            self.events.append((phase, values))

        def linearize(self):
            return self.values, self.jac

        def evaluate_trial(self, delta, trial):
            self.trials += 1
            if self.failure is not None:
                raise self.failure
            return object(), self.values + self.jac @ delta

        def accept(self, trial):
            self.promotions.append(trial)
            self.values = trial.values
            self.step += 1

    return Problem(), example.optimize, optimizer.TrialRejected


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


def test_workspace_launcher_name_does_not_shadow_scientific_workflow(problem_and_loop, monkeypatch, tmp_path):
    import sys
    from types import ModuleType

    problem, _, rejected = problem_and_loop
    launcher = ModuleType("single_stage_free_boundary_optimization")
    launcher.__file__ = str(tmp_path / "single_stage_free_boundary_optimization.py")
    monkeypatch.setitem(sys.modules, launcher.__name__, launcher)
    optimizer = importlib.import_module("optimization")
    direction = optimizer.proposal(problem.values, problem.jac, problem.parameter_scales,
                                   problem.targets, problem.constraint_scales, problem.policy)
    problem.failure = rejected("force gate failed")
    result = optimizer.backtrack(problem.values, direction, problem.targets,
                                 problem.constraint_scales, problem.policy,
                                 problem.evaluate_trial, problem.record_trial)
    assert result.candidate is None and len(result.trials) == 6
    assert problem.step == 8 and problem.promotions == []


def test_problem_uses_explicit_objective_and_constraint_callbacks(problem_and_loop, monkeypatch):
    backend = importlib.import_module("single_stage_problem")
    support = importlib.import_module("single_stage_support")
    # Construction should pass the user's definitions to the numerical layer;
    # no output directory or equilibrium solve is needed to check this boundary.
    monkeypatch.setattr(support.FreeBoundaryRun, "__init__", lambda self, args: None)
    def rows(runtime, targets, scale):
        return None

    def physical_values(state, runtime):
        return None

    def targets(initial_physical):
        return None
    problem = backend.FreeBoundaryProblem(None, rows=rows, physical_values=physical_values, targets=targets)
    assert backend._objective_functions(problem) == (physical_values, targets, rows)


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
