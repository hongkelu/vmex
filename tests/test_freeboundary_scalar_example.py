"""Exercise the actual driver callbacks with a cheap, exact implicit-root model.

No VMEX/GPU solve: these tests qualify evaluation order and state ownership only.
"""

import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace as NS
import time
import unittest

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parents[1] / "examples/three-methods-benchmark"


def harness(eager=False):
    tree = ast.parse((HERE / "_free_boundary_scalar.py").read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    functions = [
        n
        for n in run.body
        if isinstance(n, ast.FunctionDef) and n.name in ("value_and_grad", "gradient_at", "callback")
    ]
    calls = NS(solves=[], gradients=[], promoted=[], factors=[], fail_at=None, unconverged=False, evidence={})
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

        def tangent(self, accepted, cfg, delta, diagnostics=None):
            assert not self.closed and self.root is accepted
            return delta.copy()

    def pullback(root, cfg, rhs, **kw):
        calls.gradients.append(root.parameters.copy())
        return Factor(root, rhs)

    class Chart:
        size = 2

        def __call__(self, x):
            return x

    env = dict(
        np=np,
        jnp=np,
        time=time,
        calls=calls,
        chart=Chart(),
        scales=np.ones(2),
        args=NS(max_trials=100, constrained=True, accepted_steps=1, fd_ftol=1e-22),
        cfg=None,
        inp=None,
        solver=NS(resolution=None),
        ftol=1e-18,
        niter=12000,
        VmecError=ValueError,
        _solve_free_boundary_stage=solve,
        local_rows_value=rows,
        local_rows_jac=rows_jac,
        local_value=lambda state, x: (rows(state, x)[0][0], rows(state, x)[1][1]),
        objective=lambda state, x: (rows(state, x)[0][0], rows(state, x)[1][1]),
        limits=NS(inequalities=lambda x: x, physical_values=lambda s, rt: rows(s, None)[0][1:]),
        rt=None,
        metrics=lambda record: {},
        term_names=("QA",),
        write_json=lambda name, data: calls.evidence.update({name:data}),
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
    env.update(NS=NS, origin=origin, initial_u=np.zeros(2), initial=NS(parameters=np.zeros(2)), preconditioner=None)
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
