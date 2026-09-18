"""Independent QA sizing, physical feasibility, rollback and stopping checks."""

import importlib.util
import json
import sys
import os  # noqa: F401
from pathlib import Path
from vmex.core.coil_parameters import CoilParameters
import numpy as np
import pytest

CASE = Path(__file__).resolve().parents[1]
chart = CoilParameters(np.zeros((4, 3, 9)), np.ones(4), current_dofs=(1, 2, 3))


def module(name):
    spec = importlib.util.spec_from_file_location("fresh_aspect5_" + name, CASE / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


from vmex.core import projected_optimization as opt
case = module("case")
cp = module("checkpoint")
POLICY = opt.ProjectedOptions()
TARGETS = np.array([0.2, 5.0, -0.17506474574437714])
CS = case.CONSTRAINT_SCALES


def jacobian():
    jac = np.zeros((4, 111))
    jac[0, 6] = 1
    jac[1, 3] = 1
    jac[2, 4] = 1
    jac[3, 5] = 1
    return jac


def direction(values, policy=POLICY, jac=None, scales=None):
    return opt.proposal(np.asarray(values), jacobian() if jac is None else jac, np.ones(111) if scales is None else scales, TARGETS, CS, policy, motion_bounds=chart.motion_bounds)


def test_fresh_targets_tolerances_and_signed_b0():
    np.testing.assert_array_equal(np.array(list(json.loads((CASE / "case_contract.json").read_text())["targets"].values())), TARGETS)
    info = opt.constraint_state(np.array([1.0, 0, 0, 0]), TARGETS, CS, POLICY)
    np.testing.assert_allclose(info["tolerances"], [0.002, 0.05, 0.0017506474574437714])
    assert info["feasible"] and info["violation"] == 0
    contract = json.loads((CASE / "case_contract.json").read_text())
    assert contract["adjoint_solver"] == "forward_dense_jax"
    assert contract["force_tolerance"] == contract["edge_force_tolerance"] == 1e-11
    assert contract["adjoint_residual_rtol"] == 2e-5 and not contract["root_polishing"]


def test_fresh_qa_size_is_independent_of_roundoff_constraints():
    exact = direction([1, 0, 0, 0])
    tiny = direction([1, 1e-15, -1e-15, 1e-15])
    # Adaptive restoration retains roundoff-sized constraint corrections.
    # This is a floating-point comparison, not a physical acceptance tolerance.
    eps = np.finfo(np.float64).eps
    np.testing.assert_allclose(
        exact.delta, tiny.delta, rtol=32 * eps,
        atol=32 * eps * np.linalg.norm(exact.delta, ord=np.inf),
    )
    assert chart.motion_bounds(tiny.delta)[0] == pytest.approx(0.001)
    assert tiny.projected_gradient_norm == 1 and tiny.objective_directional_derivative < 0
    np.testing.assert_allclose(jacobian()[1:] @ tiny.delta, 0, atol=1e-15)


def test_fresh_infeasible_restoration_is_not_amplified():
    values = np.array([1.0, -0.46, -5.1, 0.0])
    rng = np.random.default_rng(834)
    jac = rng.normal(size=(4, 111))
    d = direction(values, jac=jac, scales=case.PARAMETER_SCALES)
    assert d.mode == "restoration"
    change = jac[1:] @ d.delta
    alpha = -change[0] / values[1]
    assert 0 < alpha <= 1
    np.testing.assert_allclose(change, -alpha * values[1:], atol=1e-12)
    assert chart.motion_bounds(d.delta)[0] <= 0.001 * (1 + 1e-12)
    assert chart.motion_bounds(d.delta)[1] <= 0.01 * (1 + 1e-12)


def test_fresh_current_only_direction_has_separate_one_percent_cap():
    jac = jacobian()
    jac[0, :] = 0
    jac[0, 0] = 100
    d = direction([1, 0, 0, 0], jac=jac)
    assert chart.motion_bounds(d.delta) == pytest.approx((0, 0.01))
    assert d.delta[0] == pytest.approx(-0.01)
    assert not opt.converged(d, d.projected_gradient_norm, POLICY)[0]


def test_fresh_continuous_coil_bound_covers_all_angles_and_both_caps():
    rng = np.random.default_rng(24)
    raw = rng.normal(size=111)
    delta = opt.limit_direction(raw, POLICY, expand=True, motion_bounds=chart.motion_bounds)
    bound, current = chart.motion_bounds(delta)
    assert bound <= 0.001 * (1 + 1e-12) and current <= 0.01 * (1 + 1e-12)
    t = np.arange(10001) / 10001
    basis = np.ones((len(t), 9))
    for k in range(1, 5):
        basis[:, 2 * k - 1] = np.sin(2 * np.pi * k * t)
        basis[:, 2 * k] = np.cos(2 * np.pi * k * t)
    motion = np.einsum("cdk,sk->csd", delta[3:].reshape(4, 3, 9), basis)
    assert np.max(np.linalg.norm(motion, axis=-1)) <= bound * (1 + 1e-12)


@pytest.mark.parametrize(
    "after,expected",
    [
        ([0.999, 0, 0, 0], True),
        ([1.0, 0, 0, 0], False),
        ([1.01, 0, 0, 0], False),
        ([0.999, 0.41, 0, 0], False),
        ([0.999, 0, 1.01, 0], False),
        ([0.999, 0, 0, 0.18], False),
        ([np.nan, 0, 0, 0], False),
    ],
)
def test_fresh_actual_qa_decrease_and_all_constraints_required(after, expected):
    before = np.array([1.0, 0, 0, 0])
    d = direction(before)
    passed, _ = opt.acceptance(before, np.asarray(after), d, 1, TARGETS, CS, POLICY)
    assert passed == expected


def test_fresh_infeasible_acceptance_prioritizes_violation_not_qa():
    before = np.array([1.0, 0.8, 0, 0])
    d = direction(before)
    assert d.mode == "restoration"
    assert opt.acceptance(before, np.array([2.0, 0.7, 0, 0]), d, 1, TARGETS, CS, POLICY)[0]
    assert not opt.acceptance(before, np.array([0.1, 0.81, 0, 0]), d, 1, TARGETS, CS, POLICY)[0]


def test_fresh_backtracking_checks_nonlinear_endpoint_and_restores_base():
    before = np.array([1.0, 0, 0, 0])
    d = direction(before)
    base = np.zeros(111)
    calls = []
    records = []

    def evaluate(delta, index):
        calls.append(delta.copy())
        actual = before + jacobian() @ delta
        actual[1] = 0.8 * (delta[6] / 0.001) ** 2
        return base + delta, actual

    result = opt.backtrack(before, d, TARGETS, CS, POLICY, evaluate, records.append, motion_bounds=chart.motion_bounds)
    assert len(calls) == 2 and not records[0]["accepted"] and records[1]["accepted"]
    np.testing.assert_array_equal(calls[1], 0.5 * calls[0])
    np.testing.assert_array_equal(result.candidate, base + calls[1])
    np.testing.assert_array_equal(base, np.zeros(111))
    assert opt.constraint_state(result.values, TARGETS, CS, POLICY)["feasible"]
    assert result.values[0] < before[0]


def test_fresh_numeric_rejections_exhaust_finite_budget_without_promotion():
    d = direction([1, 0, 0, 0])
    records = []
    calls = []

    def evaluate(delta, index):
        calls.append(delta)
        raise opt.TrialRejected("force gate failed")

    result = opt.backtrack(np.array([1.0, 0, 0, 0]), d, TARGETS, CS, POLICY, evaluate, records.append, motion_bounds=chart.motion_bounds)
    assert result.candidate is None and len(calls) == POLICY.max_trials == 6
    assert all(not r["accepted"] for r in records)
    np.testing.assert_allclose(calls[-1], calls[0] / 32)


def test_fresh_unexpected_programming_errors_do_not_become_retries():
    def evaluate(delta, index):
        raise TypeError("programming error")

    with pytest.raises(TypeError):
        opt.backtrack(np.array([1.0, 0, 0, 0]), direction([1, 0, 0, 0]), TARGETS, CS, POLICY, evaluate, lambda x: None, motion_bounds=chart.motion_bounds)


def test_fresh_small_motion_is_not_convergence():
    tiny = opt.ProjectedOptions(maximum_coil_step_m=1e-25, maximum_current_fraction_step=1e-25)
    d = direction([1, 0, 0, 0], policy=tiny)
    assert chart.motion_bounds(d.delta)[0] < 1e-20
    assert not opt.converged(d, 1.0, tiny)[0]
    result = opt.backtrack(np.array([1.0, 0, 0, 0]), d, TARGETS, CS, tiny, lambda delta, i: (delta, np.array([1.0, 0, 0, 0])), lambda x: None, motion_bounds=chart.motion_bounds)
    assert result.candidate is None


def test_fresh_convergence_requires_feasibility_and_projected_gradient():
    jac = jacobian()
    jac[0, :] = jac[1]
    d = direction([1, 0, 0, 0], jac=jac)
    assert opt.converged(d, 1.0, POLICY)[0]
    outside = direction([1, 0.8, 0, 0], jac=jac)
    assert not opt.converged(outside, 1.0, POLICY)[0]
    with pytest.raises(ValueError, match="rank deficient"):
        direction([1, 0, 0, 0], jac=np.ones((4, 111)))


def test_fresh_original_seed_is_authenticated():
    inp, builder, state, contract, hashes = case.load_case()
    assert (inp.mpol, inp.ntor, inp.nfp) == (3, 3, 2)
    assert hashes["seed_checkpoint.npz"] == "bbc25674652973ccdb2f0e1305216af6016a84c6a4ce9775afe6ca479290d9f1"
    with np.load(CASE / "inputs/seed_checkpoint.npz", allow_pickle=False) as z:
        assert int(z["accepted_step"]) == 0 and np.all(z["parameters"] == 0)
        for n in case.STATE_NAMES:
            np.testing.assert_array_equal(getattr(state, n), z[n])
