"""The level-switch rule of the least-squares free boundary's tolerance continuation (no equilibria)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from vmex.core.freeboundary_lsq import ToleranceContinuation  # noqa: E402

FINAL = 1e-15
VALUE, SENSITIVITY, COST = 2.0e-2, 3.0e-1, 3.0e-5  # an objective, its boundary sensitivity, the boundary's cost


def test_resolving_level_resolves_exactly_the_change():
    """The loosest level resolving a change has the change as its resolution (the binding term), never below FTOL."""
    rule = ToleranceContinuation(FINAL)
    for change in (1e-3, 1e-5, 1e-7):
        level = rule.resolving(change, VALUE, SENSITIVITY, COST)
        assert level >= FINAL
        if level > FINAL:
            assert rule.resolution(level, VALUE, SENSITIVITY, COST) == pytest.approx(change, rel=1e-9)
            assert rule.resolution(level * 1.01, VALUE, SENSITIVITY, COST) > change
    assert rule.resolving(1e-30, VALUE, SENSITIVITY, COST) == FINAL
    assert rule.resolving(0.0, VALUE, SENSITIVITY, COST) == FINAL
    # the boundary's term dominates a sensitive objective: a looser level than the equilibria's term alone allows
    alone = rule.resolving(1e-5, VALUE, 0.0, COST)
    assert rule.resolving(1e-5, VALUE, SENSITIVITY, COST) < alone == pytest.approx((1e-5 / (2 * VALUE)) ** 2)


def test_loosen_then_tighten_monotone_to_the_final_level():
    rule = ToleranceContinuation(FINAL)
    assert rule.level == FINAL
    assert rule.observe(VALUE, SENSITIVITY, COST) is None  # no change measured yet
    # a large step: the loosest level that resolves it (free: the values at hand resolve more)
    assert rule.observe(VALUE - 1e-4, SENSITIVITY, COST) == "loosen"
    loose = rule.level
    assert loose > FINAL
    assert rule.resolution(loose, VALUE - 1e-4, SENSITIVITY, COST) == pytest.approx(1e-4, rel=1e-9)
    # a step the level still resolves: no change
    assert rule.observe(VALUE - 2e-4, SENSITIVITY, COST) is None
    # a step below the level's resolution: tighten, to the change predicted by the steps' contraction
    value = VALUE - 2e-4 - 5e-5
    assert rule.observe(value, SENSITIVITY, COST) == "tighten"
    predicted = 5e-5 * 5e-5 / 1e-4
    assert rule.level == pytest.approx(rule.resolving(predicted, value, SENSITIVITY, COST))
    assert FINAL <= rule.level < loose
    tight = rule.level
    assert tight > FINAL
    # after a tightening the accepted point is re-solved (rebase); a large step no longer loosens
    rule.rebase(value)
    assert rule.observe(value - 1e-2, SENSITIVITY, COST) is None
    assert rule.level == tight
    # a change across levels is not measured
    rule.level = max(FINAL, tight / 10)
    assert rule.observe(value - 1e-2 - 1e-12, SENSITIVITY, COST) is None
    # the end: the deck's FTOL, once
    assert rule.finish() == "tighten" and rule.level == FINAL
    assert rule.finish() is None
    # at the final level nothing tightens further
    rule.rebase(VALUE)
    assert rule.observe(VALUE - 1e-30, SENSITIVITY, COST) is None and rule.level == FINAL


def test_restart_continues_its_level():
    rule = ToleranceContinuation(FINAL, level=1e-9, tightened=True)
    assert rule.state() == dict(level=1e-9, final=FINAL, tightened=True)
    assert ToleranceContinuation(FINAL, level=1e-20).level == FINAL  # never below the deck's FTOL
    rule.observe(VALUE, SENSITIVITY, COST)
    assert rule.observe(VALUE - 1e-1, SENSITIVITY, COST) is None  # tightened runs never loosen
    assert np.isclose(rule.level, 1e-9)


def test_model_level_never_below_the_deck_and_off_at_full_precision():
    from vmex.core.freeboundary_lsq import LeastSquaresFreeBoundaryModel

    model = LeastSquaresFreeBoundaryModel.__new__(LeastSquaresFreeBoundaryModel)
    model.cfg = type("cfg", (), {"ftol": FINAL})()
    assert model.converged_ftol == FINAL  # no level set: the deck's FTOL
    model.resolution_ftol = 1e-9
    assert model.converged_ftol == 1e-9
    model.resolution_ftol = 1e-20
    assert model.converged_ftol == FINAL
    model.resolution_ftol, model.full_precision = 1e-9, True  # a certified solve ignores the trials' level
    assert model.converged_ftol == FINAL
    assert {"resolution_ftol", "full_precision"} <= set(model._REREFERENCE_CARRIED)


def test_constraint_rows_bound_the_level_by_their_change_or_slack():
    """A constraint row near its bound (small slack) that barely moves keeps the level tight; far from its bound it
    does not constrain it; an unbounded row never does."""
    rule = ToleranceContinuation(FINAL)
    values = np.array([VALUE, 1.0e-3, 5.0])
    sens = np.array([SENSITIVITY, 1.0e-3, 1.0])
    rule.observe(values, sens, COST, slack=[1e-9, np.inf])
    step = values - np.array([1e-4, 1e-7, 1.0])
    assert rule.observe(step, sens, COST, slack=[1e-9, np.inf]) == "loosen"
    near = rule.level  # the constraint row's change 1e-7 binds, not the objective's 1e-4
    assert rule.resolution(near, step, sens, COST)[1] == pytest.approx(1e-7, rel=1e-9)
    free = ToleranceContinuation(FINAL)
    free.observe(values, sens, COST, slack=[1.0, np.inf])  # far from its bound: the objective binds
    free.observe(step, sens, COST, slack=[1.0, np.inf])
    assert free.level > near
    assert free.resolution(free.level, step, sens, COST)[0] == pytest.approx(1e-4, rel=1e-9)


def test_problem_level_switch_and_certification_drop_the_right_trials():
    """A tighter level drops the cached trials (the accepted point re-solves), a looser one keeps them; a
    certification turns full precision on and re-solves; the resolution terms are the trial's."""
    from vmex.core.freeboundary_lsq import LeastSquaresFreeBoundaryModel, LeastSquaresFreeBoundaryProblem

    model = LeastSquaresFreeBoundaryModel.__new__(LeastSquaresFreeBoundaryModel)
    model.cfg = type("cfg", (), {"ftol": FINAL})()
    model.resolution_ftol, model.full_precision = None, False
    problem = LeastSquaresFreeBoundaryProblem.__new__(LeastSquaresFreeBoundaryProblem)
    problem.model = model
    entry = dict(values=np.array([VALUE, 1.0]), boundary_sensitivity=np.array([SENSITIVITY, 2.0]), cost=COST)
    solved = []

    def record(x):
        solved.append(np.array(x))
        return problem._cache.setdefault(np.asarray(x).tobytes(), entry)

    problem._record, problem._linearize = record, lambda x, sol: None
    problem._x = lambda x: np.asarray(x, dtype=float)
    problem._cache = {b"a": entry}
    problem.set_resolution_ftol(1e-9)  # looser: kept
    assert model.converged_ftol == 1e-9 and problem._cache == {b"a": entry}
    problem.set_resolution_ftol(1e-12)  # tighter: dropped
    assert model.converged_ftol == 1e-12 and problem._cache == {}
    values, sensitivities, cost = problem.resolution_terms(np.zeros(2))
    assert np.array_equal(values, entry["values"]) and np.array_equal(sensitivities, [SENSITIVITY, 2.0])
    assert cost == COST
    problem._cache = {b"a": entry}
    assert problem.certify(np.ones(2)) is entry
    assert model.full_precision and model.converged_ftol == FINAL and list(problem._cache) == [np.ones(2).tobytes()]
    problem.set_resolution_ftol(None)
    assert model.resolution_ftol is None
