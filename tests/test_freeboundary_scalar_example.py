"""Exercise the actual driver callbacks with a cheap, exact implicit-root model.

No VMEX/GPU solve: these tests qualify evaluation order and state ownership only.
"""

import ast
import contextlib
import io
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import time
import traceback
import unittest
import weakref

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parents[1] / "examples/three-methods-benchmark"


def harness(eager=False, hybrid=False, qualified=True, convergence_policy="requested"):
    tree = ast.parse((HERE / "_free_boundary_scalar.py").read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    functions = [
        n
        for n in run.body
        if isinstance(n, ast.FunctionDef) and n.name in ("value_and_grad", "gradient_at", "callback")
    ]
    calls = NS(solves=[], gradients=[], promoted=[], factors=[], fail_at=None, unconverged=False,
               evidence={}, gradient_failure=False, failed_adjoint=None, evidence_failure=False,
               seeds=[], dense_failure=False, pullback_seeds=[], failed_workspace=None,
               workspace_alive_at_dense=None)
    origin = np.array([-1.2, 1.0])
    result = NS(iterations=1, fedge=1e-18)

    def rows(state, x):
        a, b = state
        value = (1 - a) ** 2 + 100 * (b - a * a) ** 2
        values = np.array([value, a + 10, b + 10, 10 - b])
        return values, (values, np.array([value / 2]))

    def rows_jac(state, x):
        a, b = state
        jac = np.array([[-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        values, (_, costs) = rows(state, x)
        return (jac, np.zeros_like(jac)), (values, costs)

    def solve(*_, external_field, initial_state, **kw):
        calls.solves.append((external_field.copy(), initial_state.copy()))
        if calls.fail_at is not None and np.array_equal(external_field, calls.fail_at):
            raise ValueError("synthetic failed correction")
        return NS(result=NS(state=origin + external_field, iterations=1, fedge=1e-18,
            fsqr=1e-19, fsqz=1e-19, fsql=1e-19, ier_flag=2 if calls.unconverged else 0,
            converged=not calls.unconverged), rcon0=0.0, zcon0=0.0)

    class SyntheticAdjointError(RuntimeError):
        pass

    class Seed:
        def __init__(self, root):
            self.root, self.closed = root, False
            calls.seeds.append(self)

        def close(self):
            self.closed = True

    class Factor:
        def __init__(self, root, rhs):
            self.root = root
            self.field_jacobian = rhs
            self.closed = False
            calls.factors.append(self)

        def offload_factors(self):
            pass

        def close(self):
            self.closed = True

        def preconditioner(self, **options):
            assert not self.closed
            assert options['require_adjoint_convergence'] is (convergence_policy == 'requested')
            assert options['tangent_rtol'] == 1e-11
            return Seed(self.root)

        def tangent(self, accepted, cfg, delta, diagnostics=None):
            assert not self.closed and self.root is accepted
            return delta.copy()

    def pullback(root, cfg, rhs, **kw):
        calls.gradients.append(root.parameters.copy())
        calls.pullback_seeds.append(kw.get('preconditioner'))
        if calls.gradient_failure:
            if not hybrid or kw.get('preconditioner') is not None:
                workspace = np.ones(100)
                calls.failed_workspace = weakref.ref(workspace)
                kw['diagnostics'].append(dict(accepted=False, relative_residual=float('nan')))
                raise SyntheticAdjointError("synthetic adjoint residual failure")
            calls.workspace_alive_at_dense = calls.failed_workspace() is not None
            if calls.dense_failure:
                raise RuntimeError('synthetic dense recovery failure')
        return Factor(root, rhs)

    def save_failure(out, candidate, accepted, trial, accepted_step, error):
        calls.failed_adjoint = (candidate, accepted, trial, accepted_step, error)

    def write_evidence(name, data):
        if calls.evidence_failure and name.startswith('adjoint_'):
            raise ValueError('synthetic diagnostic serialization failure')
        calls.evidence[name] = data

    class Chart:
        size = 2

        def __call__(self, x):
            return x

    env = dict(
        np=np,
        jnp=np,
        time=time,
        traceback=traceback,
        sys=__import__('sys'),
        calls=calls,
        optimization_started=1.0 if qualified else None,
        out=None,
        save_adjoint_failure=save_failure,
        chart=Chart(),
        scales=np.ones(2),
        args=NS(max_trials=100, constrained=True, accepted_steps=1, fd_ftol=1e-22,
                matrixfree_refresh_on_failure=hybrid, matrixfree_restart=100, matrixfree_max_cycles=3,
                matrixfree_rtol=1e-11, predictor_rtol=1e-11, matrixfree_convergence_policy=convergence_policy),
        cfg=None,
        inp=None,
        solver=NS(resolution=None),
        ftol=1e-18,
        niter=12000,
        VmecError=ValueError,
        AdjointSolveError=SyntheticAdjointError,
        _solve_free_boundary_stage=solve,
        local_rows_value=rows,
        local_rows_jac=rows_jac,
        local_value=lambda state, x: (rows(state, x)[0][0], rows(state, x)[1][1]),
        objective=lambda state, x: (rows(state, x)[0][0], rows(state, x)[1][1]),
        limits=NS(inequalities=lambda x: x, physical_values=lambda s, rt: rows(s, None)[0][1:]),
        rt=None,
        metrics=lambda record: {},
        term_names=("QA",),
        write_json=write_evidence,
        result=result,
        fc=NS(
            certify_free_boundary_continuation_state=lambda cfg, x, state, **kw: NS(
                parameters=x.copy(), state=state, **kw
            ),
            free_boundary_continuation_state_pullback=pullback,
        ),
        jax=NS(tree=NS(map=lambda f, *a: f(*a))),
        record_step=lambda *a: calls.promoted.append(last_ref["last"]["root"].parameters.copy()),
    )
    last_ref = {}
    body = """
def factory():
    preconditioner = initial_preconditioner
    accepted = NS(parameters=np.zeros(2), state=origin.copy(), result=result, rcon0=0., zcon0=0.)
    candidate = accepted
    accepted_linearization = candidate_linearization = None
    cycle_gradient_seconds = cycle_solve_seconds = 0.
    cycle_started = time.perf_counter()
    counts = dict(trials=0,solves=0,failed_trials=0,accepted=0)
    last = {}
    class EvaluationBudget(Exception): pass
    class AcceptedBudget(Exception): pass
"""
    for function in functions:
        body += "\n" + "\n".join("    " + line for line in ast.unparse(function).splitlines()) + "\n"
    body += "    return value_and_grad, callback, last, counts, AcceptedBudget\n"
    env.update(NS=NS, origin=origin, initial_u=np.zeros(2), initial=NS(parameters=np.zeros(2)),
               initial_preconditioner=Seed(None) if hybrid else None)
    exec(body, env)
    evaluate, accept, last, counts, stop = env["factory"]()
    last_ref["last"] = last
    return NS(
        evaluate=evaluate,
        accept=accept,
        last=last,
        counts=counts,
        stop=stop,
        calls=calls,
        value=lambda x: evaluate(x, need_gradient=eager)[0],
    )


class LazyEvaluationTests(unittest.TestCase):
    def test_failed_solver_workspace_released_before_dense_recovery(self):
        h = harness(hybrid=True)
        h.evaluate(np.zeros(2))
        h.calls.gradient_failure = True
        h.evaluate(np.array([0.1, 0.2]))
        self.assertFalse(h.calls.workspace_alive_at_dense)
        self.assertIsNone(h.calls.failed_workspace())
        # Failure evidence still owns the exception and its traceback locations.
        error = h.calls.failed_adjoint[-1]
        self.assertIsNotNone(error.__traceback__)
        self.assertIn('pullback', [frame.name for frame in traceback.extract_tb(error.__traceback__)])

    def test_dense_recovery_cannot_mask_failed_starting_qualification(self):
        h = harness(hybrid=True, qualified=False)
        h.evaluate(np.zeros(2))
        h.calls.gradient_failure = True
        with self.assertRaisesRegex(RuntimeError, 'synthetic adjoint residual failure'):
            h.evaluate(np.array([0.1, 0.2]))
        self.assertEqual(h.counts.get('dense_recovery_attempts', 0), 0)
        self.assertEqual(h.counts['accepted'], 0)

    def test_dense_recovery_refreshes_only_after_optimizer_acceptance(self):
        h = harness(hybrid=True)
        h.evaluate(np.zeros(2))
        old_seed = h.calls.seeds[0]
        h.calls.gradient_failure = True
        point = np.array([0.1, 0.2])
        h.evaluate(point)
        recovered = h.last['root']
        self.assertEqual(h.last['adjoint_backend'], 'dense_recovery')
        self.assertEqual(h.counts['dense_recoveries'], 1)
        failures = [v for k, v in h.calls.evidence.items() if k.startswith('adjoint_')
                    and v['recovered_with_dense']]
        self.assertEqual(failures[0]['checks'][0]['relative_residual'], 'nan')
        self.assertEqual(len(h.calls.seeds), 1)
        self.assertFalse(old_seed.closed)
        with self.assertRaises(h.stop):
            h.accept(point)
        self.assertTrue(old_seed.closed)
        self.assertIs(h.calls.seeds[-1].root, recovered)
        self.assertEqual(h.counts['preconditioner_refreshes'], 1)
        evidence = h.calls.evidence['preconditioner_refresh_0001.json']
        self.assertEqual(evidence['accepted_step'], 1)

    def test_unaccepted_dense_recovery_does_not_replace_seed(self):
        h = harness(hybrid=True)
        h.evaluate(np.zeros(2))
        old_seed = h.calls.seeds[0]
        h.calls.gradient_failure = True
        h.evaluate(np.array([0.1, 0.2]))
        abandoned = h.calls.factors[-1]
        h.calls.gradient_failure = False
        h.evaluate(np.array([0.05, 0.1]))
        self.assertTrue(abandoned.closed)
        self.assertIs(h.calls.pullback_seeds[-1], old_seed)
        self.assertFalse(old_seed.closed)
        self.assertEqual(len(h.calls.seeds), 1)
        self.assertEqual(h.counts['accepted'], 0)

    def test_dense_recovery_failure_keeps_anchor_and_original_cause(self):
        h = harness(hybrid=True)
        h.evaluate(np.zeros(2))
        anchor = h.calls.factors[0]
        h.calls.gradient_failure = h.calls.dense_failure = True
        with self.assertRaisesRegex(RuntimeError, 'synthetic dense recovery failure') as caught:
            h.evaluate(np.array([0.1, 0.2]))
        self.assertIn('synthetic adjoint residual failure', str(caught.exception.__cause__))
        self.assertFalse(anchor.closed)
        self.assertFalse(h.calls.seeds[0].closed)
        self.assertEqual(h.counts['accepted'], 0)
        self.assertIsNone(h.last['gradient'])

    def test_diagnostic_write_failure_preserves_original_adjoint_error(self):
        h = harness()
        h.evaluate(np.zeros(2))
        h.value(np.array([0.1, 0.2]))
        h.calls.gradient_failure = h.calls.evidence_failure = True
        with self.assertRaisesRegex(RuntimeError, 'synthetic adjoint residual failure') as caught:
            h.evaluate(np.array([0.1, 0.2]))
        self.assertIn('synthetic diagnostic serialization failure', caught.exception.__notes__[0])
        self.assertEqual(h.counts['accepted'], 0)

    def test_failed_adjoint_saves_trial_with_last_accepted_anchor(self):
        h = harness()
        h.evaluate(np.zeros(2))
        anchor = h.calls.factors[0].root
        trial = np.array([0.1, 0.2])
        h.value(trial)
        h.calls.gradient_failure = True
        with self.assertRaisesRegex(RuntimeError, "synthetic adjoint residual failure"):
            h.evaluate(trial)
        candidate, saved_anchor, _, step, _ = h.calls.failed_adjoint
        self.assertIs(saved_anchor, anchor)
        np.testing.assert_array_equal(candidate.parameters, trial)
        self.assertEqual(step, 0)
        self.assertEqual(h.counts['accepted'], 0)
        self.assertEqual(h.counts['failed_adjoint_trials'], 1)
        self.assertFalse(h.calls.factors[0].closed)
        self.assertIsNone(h.last['gradient'])

    def test_values_and_constraints_share_root_then_gradient_reuses_it(self):
        h = harness()
        h.evaluate(np.zeros(2))
        point = np.array([0.1, 0.2])
        h.value(point)
        root = h.last["root"]
        h.value(point)
        self.assertEqual(len(h.calls.solves), 1)
        self.assertEqual(len(h.calls.gradients), 1)
        h.evaluate(point)
        h.evaluate(point)
        self.assertIs(h.last["root"], root)
        self.assertEqual(len(h.calls.solves), 1)
        self.assertEqual(len(h.calls.gradients), 2)

    def test_rejected_trial_never_becomes_next_predictor_anchor(self):
        h = harness()
        h.evaluate(np.zeros(2))
        h.value(np.array([0.2, 0.3]))
        h.value(np.array([0.1, 0.15]))
        for x, prediction in h.calls.solves:
            np.testing.assert_array_equal(prediction, np.array([-1.2, 1.0]) + x)
        self.assertEqual(h.counts["accepted"], 0)
        self.assertEqual(len(h.calls.gradients), 1)
        self.assertFalse(h.calls.factors[0].closed)

    def test_uncertified_gradient_cannot_be_promoted(self):
        h = harness()
        h.evaluate(np.zeros(2))
        h.value(np.array([0.1, 0.1]))
        with self.assertRaisesRegex(RuntimeError, "before its certified gradient"):
            h.accept(np.array([0.1, 0.1]))
        self.assertEqual(h.counts["accepted"], 0)

    def test_failed_trial_preserves_previous_finite_root_and_factors(self):
        h = harness()
        h.evaluate(np.zeros(2))
        p = np.array([0.1, 0.1])
        h.evaluate(p)
        root, factor = h.last["root"], h.calls.factors[-1]
        h.calls.fail_at = np.array([0.2, 0.2])
        self.assertTrue(np.isinf(h.value(h.calls.fail_at)))
        self.assertIs(h.last["root"], root)
        self.assertFalse(factor.closed)
        h.evaluate(p)
        with self.assertRaises(h.stop):
            h.accept(p)
        self.assertIs(h.calls.factors[-1].root, root)

    def test_finite_difference_does_not_contaminate_cached_trial(self):
        h = harness()
        h.evaluate(np.zeros(2))
        p = np.array([0.1, 0.1])
        h.value(p)
        root = h.last["root"]
        h.evaluate(np.array([0.01, 0.01]), derivative=False)
        self.assertIs(h.last["root"], root)
        h.evaluate(p)
        self.assertIs(h.calls.factors[-1].root, root)
        self.assertEqual(len(h.calls.gradients), 2)

    def test_failed_fd_preserves_the_cached_root_and_reports_original_error(self):
        h = harness()
        h.evaluate(np.zeros(2))
        root = h.last["root"]
        h.calls.fail_at = np.array([0.01, 0.01])
        with self.assertRaisesRegex(RuntimeError, "finite-difference equilibrium did not converge"):
            h.evaluate(h.calls.fail_at, derivative=False)
        self.assertIs(h.last["root"], root)
        self.assertEqual(h.counts["accepted"], 0)

    def test_unconverged_fd_saves_force_evidence_and_cannot_be_promoted(self):
        h = harness()
        h.evaluate(np.zeros(2))
        root = h.last["root"]
        h.calls.unconverged = True
        with self.assertRaisesRegex(RuntimeError, "equilibrium did not converge"):
            h.evaluate(np.array([.01,.01]), derivative=False)
        self.assertIs(h.last["root"], root)
        self.assertEqual(h.counts["accepted"], 0)
        evidence = next(v for k,v in h.calls.evidence.items() if k.startswith('failed_solve_'))
        self.assertEqual(evidence['fedge'], 1e-18)
        self.assertEqual(evidence['force_tolerance'], 1e-22)
        self.assertTrue(evidence['finite_difference'])

    def test_slsqp_same_first_accepted_step_fewer_gradients(self):
        runs = []
        for eager in (True, False):
            h = harness(eager)

            def jac(x):
                _, gradient = h.evaluate(x)
                h.accept(x)
                return gradient

            def con(x):
                h.value(x)
                return h.last["constraints"].copy()

            def con_jac(x):
                h.evaluate(x)
                return h.last["constraint_jacobian"].copy()

            with self.assertRaises(h.stop):
                minimize(
                    h.value,
                    np.zeros(2),
                    jac=jac,
                    method="SLSQP",
                    constraints=[dict(type="ineq", fun=con, jac=con_jac)],
                    options=dict(maxiter=2, ftol=1e-10),
                )
            runs.append(h)
        old, new = runs
        np.testing.assert_array_equal(old.calls.promoted, new.calls.promoted)
        np.testing.assert_array_equal([x for x, _ in old.calls.solves], [x for x, _ in new.calls.solves])
        self.assertLess(len(new.calls.gradients), len(old.calls.gradients))
        self.assertEqual(len(new.calls.gradients), 2)  # initial and accepted only
        print(
            f"Synthetic SLSQP: same first accepted step; gradients {len(old.calls.gradients)} -> {len(new.calls.gradients)}"
        )


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        result = unittest.main(exit=False, verbosity=2)
    print("\n".join(line for line in stream.getvalue().splitlines() if line.startswith("Synthetic")))
    raise SystemExit(not result.result.wasSuccessful())


def load_checks():
    import importlib.util

    spec = importlib.util.spec_from_file_location("scalar_gradient_check", HERE / "_scalar_gradient_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.centered_checks


def test_centered_checks_match_analytic_values():
    checks = load_checks()

    def evaluate(x):
        return dict(value=float(x @ x), constraints=[float(x.sum()), float(-x.sum())])

    x = np.array([1.0, 2.0])
    direction = np.array([0.3, -0.7])
    rows, endpoints = checks(
        evaluate, x, direction, 2 * x @ direction, np.array([direction.sum(), -direction.sum()]), save=lambda r: None
    )
    assert len(endpoints) == 4
    assert all(r["relative_error"] < 1e-10 and max(r["constraint_relative_errors"]) < 1e-10 for r in rows)


def test_centered_check_failure_preserves_plus_endpoint_and_original_cause():
    import pytest

    checks = load_checks()
    saved = []

    def evaluate(x):
        if x[0] < 0:
            raise RuntimeError("MORE ITERATIONS REQUIRED")
        return dict(value=float(x[0]), constraints=[float(x[0])])

    with pytest.raises(RuntimeError, match="h=0.003, sign=-1: MORE ITERATIONS REQUIRED") as error:
        checks(evaluate, np.zeros(1), np.ones(1), 1.0, np.ones(1), save=saved.append)
    assert str(error.value.__cause__) == "MORE ITERATIONS REQUIRED"
    assert saved[0]["passed"] is False and saved[0]["failure"]["sign"] == -1
    assert len(saved[0]["endpoints"]) == 1 and saved[0]["checks"] == []


def test_nonfinite_fd_endpoint_cannot_pass():
    import pytest

    saved = []
    with pytest.raises(RuntimeError, match="nonfinite"):
        load_checks()(lambda x: dict(value=np.nan), np.zeros(1), np.ones(1), 1.0, None, save=saved.append)
    assert saved[0]["passed"] is False and saved[0]["endpoints"] == []


def test_failure_checkpoint_roundtrip_and_hashes(tmp_path):
    spec = importlib.util.spec_from_file_location("scalar_failure_driver", HERE / "_scalar_resume.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
    def root(offset):
        return NS(parameters=np.array([offset, 2.0]), rcon0=np.full((2, 3), offset),
                  zcon0=np.full((2, 3), -offset),
                  state=NS(**{name:np.full((2, 4), i+offset) for i,name in enumerate(names)}))
    anchor, trial = root(1.0), root(1.5)
    module.save_adjoint_failure(tmp_path, trial, anchor, 15, 8, RuntimeError("residual gate"))
    metadata = json.loads((tmp_path / "failed_adjoint_0015.json").read_text())
    assert metadata['accepted_step'] == 8 and not metadata['trial_accepted']
    for label, expected in (("anchor",anchor),("trial",trial)):
        item = metadata['checkpoints'][label]
        path = tmp_path / item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved['parameters'],expected.parameters)
            np.testing.assert_array_equal(saved['rcon0'],expected.rcon0)
            np.testing.assert_array_equal(saved['zcon0'],expected.zcon0)
            for name in names:
                np.testing.assert_array_equal(saved[name],getattr(expected.state,name))
        arrays, provenance = module.load_seed_checkpoint(path, item['sha256'])
        np.testing.assert_array_equal(arrays['parameters'], expected.parameters)
        assert provenance['sha256'] == item['sha256']
        assert 'fresh SLSQP state' in provenance['optimizer_state']


def test_diagnostic_seed_rejects_wrong_hash_missing_baselines_and_nonfinite(tmp_path):
    import pytest

    spec = importlib.util.spec_from_file_location("scalar_seed_driver", HERE / "_scalar_resume.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / 'seed.npz'
    arrays = {name: np.ones((2, 3)) for name in
              ('parameters', 'rcon0', 'zcon0', 'R_cos', 'R_sin', 'Z_cos', 'Z_sin', 'L_cos', 'L_sin')}
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        module.load_seed_checkpoint(path, '0'*64)
    del arrays['rcon0']
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='constraint baselines'):
        module.load_seed_checkpoint(path, hashlib.sha256(path.read_bytes()).hexdigest())
    arrays['rcon0'] = np.full((2, 3), np.nan)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='nonfinite'):
        module.load_seed_checkpoint(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_dense_recovery_preserves_residual_acceptance_policy():
    h = harness(hybrid=True, convergence_policy="residual")
    h.evaluate(np.zeros(2))
    h.calls.gradient_failure = True
    point = np.array([0.1, 0.2])
    h.evaluate(point)
    import pytest

    with pytest.raises(h.stop):
        h.accept(point)
    assert h.counts["preconditioner_refreshes"] == 1


def test_scalar_tolerance_defaults_and_invalid_options(monkeypatch):
    import argparse
    from dataclasses import replace
    import pytest
    import sys

    monkeypatch.syspath_prepend(str(HERE))
    from free_boundary_single_stage_optimization_scalar import settings

    tree = ast.parse((HERE / "_free_boundary_scalar.py").read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    end = next(i for i, n in enumerate(run.body)
               if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "out" for t in n.targets))
    run.body = run.body[:end] + [ast.Return(value=ast.Name(id="args", ctx=ast.Load()))]
    env = dict(argparse=argparse, Path=Path, time=time, replace=replace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[run], type_ignores=[])), "driver options", "exec"), env)

    def parse(*options):
        monkeypatch.setattr(sys, "argv", ["scalar", *options])
        return env["run"](HERE / "free_boundary_single_stage_optimization_scalar.py", settings, None, None)

    args = parse("--constrained")
    assert args.adjoint == "dense"
    assert (args.ftol, args.verify_ftol, args.fd_ftol) == (1e-11, 1e-15, 1e-20)
    assert (args.matrixfree_rtol, args.predictor_rtol, args.adjoint_residual_rtol) == (1e-11, 1e-11, 1e-9)
    assert args.matrixfree_convergence_policy == "residual"
    for option in ("--matrixfree-rtol", "--predictor-rtol", "--adjoint-residual-rtol"):
        for value in ("nan", "inf", "0", "1"):
            with pytest.raises(SystemExit) as error:
                parse(option, value)
            assert error.value.code == 2
    with pytest.raises(SystemExit):
        parse("--adjoint", "matrixfree", "--no-check-gradient")


def test_free_settings_match_fixed_control():
    import sys
    # Load only the small settings and diagnostics modules, not the fixed run.
    previous = sys.path.copy()
    try:
        sys.path.insert(0, str(HERE))
        from free_boundary_single_stage_optimization_scalar import settings
        from _scalar_diagnostics import check_control_settings
        matched = check_control_settings(settings, HERE / "single_stage_optimization_scalar.py")
    finally:
        sys.path[:] = previous
    assert matched["n_coils"] == settings.n_coils
    assert "gradient_check_ftol" not in matched
