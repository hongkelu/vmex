"""Independent coupling, physical profile, and native residual construction checks."""

from pathlib import Path
import sys
from dataclasses import replace
import json
import numpy as np
import pytest
import jax
import jax.numpy as jnp

CASE = Path(__file__).resolve().parents[1] / "examples/finite-beta-bootstrap-benchmark"
sys.path.insert(0, str(CASE))
import bootstrap_settings as S
from linear_root import LinearRoot, Options, RootFailure, SlowProgress, ReuseCost
import case as entry
from physics import kinetic_profiles, pressure_input, EquilibriumCurrentRoot


@pytest.mark.usefixtures("_module_jit_enabled")
@pytest.mark.parametrize("assembly", ["host", "device"])
def test_device_coupled_root_batch_derivative_and_executable_reuse(assembly):
    from device_root import DeviceRoot, _krylov

    def residual(y, x):
        z, q = y
        return jnp.array([z + .2*z*z - q - x[0], q - .3*z*z - .4*x[0]])

    def rows(y, x):
        a = y[0]**2 + 2*y[1] + x[0]
        return jnp.array([a, 2*a, -a, x[0]])

    events = []
    engine = DeviceRoot(residual, dense_assembly=assembly, rhs_batch_size=2,
                        options=Options(atol=1e-13, max_dofs=8, batch=2, saved_linearization=True,
                                        newton_rtol=1e-6, newton_gate=1e-5),
                        event=lambda **kw: events.append(kw))
    x = np.array([.2])
    root = engine.solve(np.array([.1, .1]), x)
    _, jac = engine.derivative(rows, root, x)
    sizes = _krylov._cache_size()
    values = []
    for sign in (-1, 1):
        point = x + sign*1e-5
        corrected = engine.solve(root.copy(), point)
        values.append(np.asarray(rows(corrected, point)))
        engine.derivative(rows, corrected, point)
    np.testing.assert_allclose(jac[:, 0], (values[1]-values[0])/2e-5, rtol=2e-8, atol=1e-9)
    assert _krylov._cache_size() == sizes
    assert all(max(e['relative_residuals']) <= e['residual_gate']
               for e in events if e['event'] == 'linear_batch_solve')
    assert max(e['rows'] for e in events if e['event'] == 'linear_batch_solve') == 2


@pytest.mark.usefixtures("_module_jit_enabled")
def test_device_actual_residual_recovery_and_rejected_trial_ownership():
    from device_root import DeviceRoot
    events = []
    engine = DeviceRoot(lambda y, x: (2+x[0])*y-x, options=Options(max_dofs=3, batch=1,
                        saved_linearization=True), event=lambda **kw: events.append(kw))
    x, y = np.array([.2]), np.array([.2/2.2])
    engine.rebuild(y, x)
    engine.initialize_accepted(y, x, remaining=10)
    accepted = engine.factors
    engine.begin_trial(np.array([.4]))
    engine.factors = (np.full((1, 1), np.nan), np.array([0], dtype=np.int32))
    value = engine.linear(y, np.array([.4]), np.array([1.0]), transpose=True)
    np.testing.assert_allclose(value, [1/2.4], atol=1e-13)
    assert any(e['event'] == 'dense_recovery' for e in events)
    assert engine.factors is not accepted and engine._accepted[0] is accepted
    engine.retain_trial(np.array([.4]))
    engine.discard_trial()
    assert engine.factors is accepted and not engine._trial
    assert engine._refresh.remaining == 10


@pytest.mark.usefixtures("_module_jit_enabled")
@pytest.mark.parametrize("assembly", ["host", "device"])
def test_trial_dense_budget_rejects_before_second_rebuild_and_keeps_strict_recovery(monkeypatch, assembly):
    import device_root as module

    events = []
    matrix = lambda x: jnp.diag(jnp.array([2+x[0], 3+2*x[0]]))
    engine = module.DeviceRoot(lambda y, x: matrix(x) @ y, dense_assembly=assembly,
        options=Options(max_dofs=4, batch=2, saved_linearization=True, trial_dense_rebuilds=1),
        event=lambda **kw: events.append(kw))
    y, x0, x1, x2 = np.zeros(2), np.array([0.]), np.array([10.]), np.array([20.])
    rhs = np.array([1., 2.])
    engine.rebuild(y, x0)
    engine.initialize_accepted(y, x0, remaining=10)
    accepted = engine.factors
    engine.begin_trial(x2)
    # Deliberately fail Krylov; actual AD-defect checks must force checked LU
    # recovery once, and request subdivision before a second dense rebuild.
    monkeypatch.setattr(engine, "_warm", lambda *a, **k: None)
    monkeypatch.setattr(module, "_krylov", lambda tangent, factors, rhs, **kw:
        (jnp.zeros_like(rhs), jnp.ones(rhs.shape[0], dtype=int), jnp.ones(rhs.shape[0]),
         jnp.zeros(rhs.shape[0], dtype=bool)))
    with pytest.raises(SlowProgress, match="dense rebuild budget"):
        with engine.continuation_segment():
            value = engine.linear(y, x1, rhs, newton=True)
            np.testing.assert_allclose(np.asarray(matrix(x1)) @ value, rhs, atol=1e-13)
            engine.linear(y, x2, rhs, newton=True)
    assert sum(e['event'] == 'dense_factor' for e in events) == 2  # initial + one trial
    assert engine.factors is accepted and engine._accepted[0] is accepted
    assert engine._step_seconds > 0 and engine._segment_dense_builds is None
    assert engine._refresh.remaining == 10
    # A new smaller segment gets its own recovery budget. Strict adjoints
    # still recover even after that budget has been spent.
    with engine.continuation_segment():
        engine.linear(y, x1, rhs, newton=True)
        value = engine.linear(y, x2, rhs, transpose=True)
        np.testing.assert_allclose(np.asarray(matrix(x2)).T @ value, rhs, atol=1e-13)
    assert engine.factors is not accepted and engine._accepted[0] is accepted
    engine.discard_trial()
    assert engine.factors is accepted


@pytest.mark.parametrize("stagnation_steps", [2, 3])
def test_strong_damping_subdivides_earlier_without_relaxing_root(stagnation_steps):
    events = []
    engine = LinearRoot(lambda y, x: y - 1, valid=lambda y, x: y[0] <= .02,
        options=Options(max_dofs=2, stagnation_steps=stagnation_steps),
        event=lambda **kw: events.append(kw))
    seed = np.zeros(1)
    with pytest.raises(SlowProgress, match="strong damping"):
        engine.solve(seed, np.zeros(1))
    assert sum(e['event'] == 'root_polish' for e in events) == stagnation_steps
    assert not any(e['event'] == 'root_converged' for e in events)
    np.testing.assert_array_equal(seed, [0.])


def test_damped_but_productive_newton_still_reaches_strict_root():
    engine = LinearRoot(lambda y, x: jnp.arctan(y),
        options=Options(max_dofs=2, stagnation_steps=2, atol=1e-13))
    root = engine.solve(np.array([2.]), np.zeros(1))
    assert np.linalg.norm(np.arctan(root)) <= 1e-13
    assert engine.last_solve['minimum_alpha'] < 1


@pytest.mark.usefixtures("_module_jit_enabled")
def test_device_acceptance_refresh_parity_is_transactional(monkeypatch):
    from device_root import DeviceRoot
    engine = DeviceRoot(lambda y, x: 2*y-x, options=Options(max_dofs=3, batch=1,
                        saved_linearization=True))
    x0, x1 = np.array([.2]), np.array([.3])
    engine.rebuild(x0/2, x0)
    engine.initialize_accepted(x0/2, x0, remaining=10)
    accepted = engine.factors
    engine.begin_trial(x1)
    engine._step_seconds = 10
    engine.retain_trial(x1)
    engine._refresh.warmup = 0
    engine._refresh.costs = [10., 10.]
    engine._refresh.best_seconds = .01
    engine._refresh.dense_seconds = .01
    rows = lambda y, x: y
    monkeypatch.setattr(engine, 'derivative', lambda *a: (np.array([.15]), np.array([[100.]])))
    with pytest.raises(RootFailure, match='changed derivative'):
        engine.accept_trial(x1/2, x1, rows, np.array([[.5]]))
    assert engine._accepted[0] is accepted and engine._accepted_key == x0.tobytes()
    assert engine._refresh.remaining == 10
    monkeypatch.undo()
    jac = engine.accept_trial(x1/2, x1, rows, np.array([[.5]]))
    np.testing.assert_allclose(jac, [[.5]], atol=1e-13)
    assert engine._accepted_key == x1.tobytes() and engine._accepted[0] is not accepted
    assert engine._refresh.remaining == 9


def test_coupled_current_total_derivative_matches_reconverged_endpoints():
    # Equilibrium z depends on bootstrap current q; q in turn depends on z and x.
    def residual(y, x):
        z, q = y
        return jnp.array([z + 0.2 * z * z - q - x[0], q - 0.3 * z * z - 0.4 * x[0]])

    solver = LinearRoot(residual, options=Options(atol=1e-13, max_dofs=10))
    x = np.array([0.2])
    root = solver.solve(np.array([0.1, 0.1]), x)

    def rows(y, x):
        return jnp.array([y[0] ** 2 + 2 * y[1] + x[0], y[1]])

    values, jac = solver.derivative(rows, root, x)
    fd = []
    for sign in (-1, 1):
        point = x + sign * 1e-5
        fd.append(np.asarray(rows(solver.solve(root.copy(), point), point)))
    np.testing.assert_allclose(jac[:, 0], (fd[1] - fd[0]) / 2e-5, rtol=2e-8, atol=1e-9)
    # Freezing q omits feedback and demonstrably gives the wrong gradient.
    frozen_current_grad = 2 * root[0] / (1 + 0.4 * root[0]) + 1
    assert abs(jac[0, 0] - frozen_current_grad) > 0.5


def test_stale_lu_recovery_and_nonfinite_rejection():
    events = []
    solver = LinearRoot(
        lambda y, x: jnp.array([(1 + x[0]) * y[0] - x[0]]),
        options=Options(max_dofs=4),
        event=lambda **kw: events.append(kw),
    )
    root = solver.solve(np.array([0.0]), np.array([0.3]))
    solver.factors = (np.array([[np.nan]]), np.array([0], dtype=np.int32))
    result = solver.linear(root, np.array([0.3]), np.array([1.0]), transpose=True)
    np.testing.assert_allclose(result, [1 / 1.3])
    assert sum(e["event"] == "dense_factor" for e in events) >= 2
    with pytest.raises(RootFailure, match="nonfinite"):
        solver.linear(root, np.array([0.3]), np.array([np.nan]))


def test_newton_starts_from_copy_and_rejects_invalid_geometry():
    solver = LinearRoot(lambda y, x: y - x, valid=lambda y, x: bool(np.all(y > 0)), options=Options(max_dofs=3))
    seed = np.array([1.0])
    x = np.array([2.0])
    np.testing.assert_allclose(solver.solve(seed, x), x)
    np.testing.assert_array_equal(seed, [1.0])
    with pytest.raises(RootFailure, match="geometry"):
        solver.solve(np.array([-1.0]), x)


def test_reuse_cost_ignores_warmup_and_separates_transpose():
    cost = ReuseCost(dense_seconds=100, horizon=12)
    assert not any(cost.observe(v, False) for v in (200, 150, 1, 1, 1, 15))
    assert cost.observe(15, False)
    assert not any(cost.observe(v, True) for v in (300, 250, 40, 40, 40))
    disabled = ReuseCost(dense_seconds=100, horizon=0)
    assert not any(disabled.observe(v, False) for v in (200, 150, 1, 1, 1, 15, 15))


@pytest.mark.parametrize("transpose", [False, True])
def test_slow_reuse_refresh_preserves_current_tangent_gate(monkeypatch, transpose):
    import linear_root as module

    matrix = jnp.array([[2.0, 0.4], [-0.2, 3.0]])
    events = []
    solver = LinearRoot(
        lambda y, x: (matrix + x[0] * jnp.eye(2)) @ y, options=Options(max_dofs=4), event=lambda **kw: events.append(kw)
    )
    solver.rebuild(np.zeros(2), np.zeros(1))
    solver.cost.dense_seconds = 1.0
    original = module.gmres

    def stalled(*args, **kwargs):
        assert "callback" in kwargs
        # Trigger the same bounded refresh decision as a costly JVP sequence.
        solver.cost.dense_seconds = 1
        clock = module.time.monotonic
        monkeypatch.setattr(module.time, "monotonic", lambda: clock() + 20)
        try:
            kwargs["callback"](1.0)
        finally:
            monkeypatch.setattr(module.time, "monotonic", clock)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "gmres", stalled)
    rhs = np.array([0.7, -0.4])
    result = solver.linear(np.zeros(2), np.array([0.5]), rhs, transpose=transpose)
    current = np.asarray(matrix) + 0.5 * np.eye(2)
    np.testing.assert_allclose((current.T if transpose else current) @ result, rhs, atol=1e-13)
    assert any(e.get("reason") == "krylov_time_budget" for e in events)
    assert sum(e["event"] == "dense_factor" for e in events) == 2
    assert events[-1]["relative_residual"] <= solver.options.gate


def test_newton_extends_only_rapid_convergence_without_loosening_root():
    events = []

    def residual(y, x):
        return y * y - x

    seed, x = np.array([1.1]), np.array([1.0])
    strict = LinearRoot(residual, options=Options(atol=1e-14, newton_steps=2, extra_newton_steps=0))
    with pytest.raises(SlowProgress, match="iteration budget"):
        strict.solve(seed, x)
    solver = LinearRoot(
        residual,
        options=Options(atol=1e-14, newton_steps=2, extra_newton_steps=3),
        event=lambda **kw: events.append(kw),
    )
    root = solver.solve(seed, x)
    assert np.linalg.norm(residual(root, x)) <= 1e-14
    assert 2 < solver.last_solve["iterations"] <= 5
    assert any(e["event"] == "newton_extension" for e in events)
    np.testing.assert_array_equal(seed, [1.1])
    slow = LinearRoot(lambda y, x: jnp.exp(y), options=Options(atol=1e-14, newton_steps=2, extra_newton_steps=3))
    with pytest.raises(SlowProgress, match="iteration budget"):
        slow.solve(np.array([0.0]), x)


def test_first_adjoint_compilation_does_not_trigger_stale_lu_budget(monkeypatch):
    import linear_root as module

    events, clock = [], [0.0]
    matrix = np.array([[2.0, 0.4], [-0.2, 3.0]])
    solver = LinearRoot(
        lambda y, x: jnp.asarray(matrix) @ y,
        options=Options(max_dofs=4, separate_action_warmup=True),
        event=lambda **kw: events.append(kw),
    )
    solver.rebuild(np.zeros(2), np.zeros(1))
    solver.cost.dense_seconds = 1.0
    calls = []

    def first_use_compile(y, x, v):
        if not calls:
            clock[0] += 40.0  # exceeds the 10-second reused-solve budget
        calls.append(True)
        return matrix.T @ np.asarray(v)

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(solver, "_vjp", first_use_compile)
    rhs = np.array([0.3, -0.5])
    for _ in range(2):
        value = solver.linear(np.zeros(2), np.zeros(1), rhs, transpose=True)
        np.testing.assert_allclose(matrix.T @ value, rhs, atol=1e-13)
    warmups = [e for e in events if e["event"] == "linear_action_warmup"]
    assert len(warmups) == 1 and warmups[0]["seconds"] == 40.0
    assert not any(e["event"] == "preconditioner_refresh" for e in events)
    assert sum(e["event"] == "dense_factor" for e in events) == 1


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("saved_krylov", [False, True])
def test_saved_tangent_tracks_state_and_design_without_reusing_a_stale_operator(transpose, saved_krylov):
    def residual(y, x):
        return jnp.array([y[0] ** 2 + x[0] * y[1], 0.2 * y[0] + (2 + x[0]) * y[1]])

    solver = LinearRoot(
        residual,
        options=Options(
            max_dofs=4,
            batch=3,
            saved_linearization=True,
            saved_krylov_actions=saved_krylov,
            separate_action_warmup=True,
        ),
    )
    y, x, rhs = np.array([1.0, 0.2]), np.array([0.3]), np.array([0.7, -0.4])
    for change in ("none", "state", "design"):
        if change == "state":
            y[0] = 1.7
        if change == "design":
            x[0] = 0.6
        result = solver.linear(y, x, rhs, transpose=transpose)
        matrix = np.asarray(jax.jacfwd(residual)(jnp.asarray(y), jnp.asarray(x)))
        np.testing.assert_allclose((matrix.T if transpose else matrix) @ result, rhs, rtol=1e-9, atol=1e-12)


def test_same_point_saved_factor_serves_multiple_rhs_without_krylov(monkeypatch):
    import linear_root as module

    solver = LinearRoot(
        lambda y, x: jnp.array([2 * y[0] + y[1], y[0] + 3 * y[1]]),
        options=Options(max_dofs=4, saved_linearization=True),
    )
    y, x = np.zeros(2), np.zeros(1)
    solver.rebuild(y, x)

    def unexpected(*args, **kwargs):
        pytest.fail("an unchanged exact-point LU should not need Krylov iteration")

    monkeypatch.setattr(module, "gmres", unexpected)
    for transpose, rhs in [(False, np.array([0.7, -0.2])), (True, np.array([-0.3, 0.9]))]:
        value = solver.linear(y, x, rhs, transpose=transpose)
        np.testing.assert_allclose(np.array([[2, 1], [1, 3]]) @ value, rhs, atol=1e-13)


def test_saved_operator_cannot_bypass_actual_residual_gate(monkeypatch):
    import linear_root as module

    events = []
    solver = LinearRoot(
        lambda y, x: jnp.array([(2 + x[0]) * y[0] + y[1], y[0] + (3 + x[0]) * y[1]]),
        options=Options(max_dofs=4, saved_linearization=True, saved_krylov_actions=True),
        event=lambda **kw: events.append(kw),
    )
    solver.rebuild(np.zeros(2), np.zeros(1))
    # A corrupted cached action solves the wrong Krylov system exactly. The
    # independent check must detect it and recover with the current matrix.
    monkeypatch.setattr(module, "_tangent_apply", lambda tangent, vector: 3 * vector)
    rhs = np.array([0.7, -0.2])
    value = solver.linear(np.zeros(2), np.array([0.8]), rhs)
    np.testing.assert_allclose(np.array([[2.8, 1.0], [1.0, 3.8]]) @ value, rhs, atol=1e-12)
    checks = [e for e in events if e["event"] == "linear_solve"]
    assert checks[0]["relative_residual"] > solver.options.gate
    assert checks[-1]["relative_residual"] <= solver.options.gate
    assert sum(e["event"] == "dense_factor" for e in events) == 2


def test_newton_forcing_does_not_relax_root_predictor_or_adjoint_gates():
    events = []

    def residual(y, x):
        z, q = y
        return jnp.array([z + 0.2 * z * z - q - x[0], q - 0.3 * z * z - 0.4 * x[0]])

    solver = LinearRoot(
        residual,
        options=Options(max_dofs=4, atol=1e-13, saved_linearization=True, newton_rtol=1e-6, newton_gate=1e-5),
        event=lambda **kw: events.append(kw),
    )
    x = np.array([0.2])
    root = solver.solve(np.array([0.1, 0.1]), x)
    assert np.linalg.norm(solver.residual(root, x)) <= 1e-13
    solver.predict(root, x, x + 1e-5)

    def rows(y, x):
        return jnp.array([y[0] ** 2 + 2 * y[1] + x[0]])

    _, jac = solver.derivative(rows, root, x)
    values = [rows(solver.solve(root, x + sign * 1e-5), x + sign * 1e-5)[0] for sign in (-1, 1)]
    np.testing.assert_allclose(jac[0, 0], (values[1] - values[0]) / 2e-5, rtol=2e-8, atol=1e-9)
    for purpose, gate in [("newton", 1e-5), ("tangent", 1e-9), ("adjoint", 1e-9)]:
        checks = [e for e in events if e["event"] == "linear_solve" and e["purpose"] == purpose]
        assert checks and all(e["residual_gate"] == gate and e["relative_residual"] <= gate for e in checks)
    with pytest.raises(ValueError, match="adjoint"):
        solver.linear(root, x, np.ones(2), transpose=True, newton=True)


def test_pressure_matches_frozen_ne_te_ti_and_has_zero_edge():
    import vmex as vj

    inp = vj.VmecInput.from_file(entry.REFERENCE / "input.rotating_ellipse")
    prepared = pressure_input(inp, 437.0)
    profiles = kinetic_profiles(437.0)
    s = np.linspace(0, 1, 31)
    poly = np.polynomial.polynomial.polyval
    from vmex.core.bootstrap import ELEMENTARY_CHARGE

    physical = (
        ELEMENTARY_CHARGE * poly(s, profiles.ne_coeffs) * (poly(s, profiles.Te_coeffs) + poly(s, profiles.Ti_coeffs))
    )
    np.testing.assert_allclose(prepared.pres_scale * poly(s, prepared.am), physical, atol=1e-10, rtol=1e-12)
    assert prepared.phiedge == inp.phiedge and physical[-1] == 0
    assert S.BETA_AXIS_INITIAL == 0.03 and prepared.ncurr == 1


def test_native_current_survives_deck_roundtrip_and_matches_both_runtime_paths(tmp_path):
    import vmex as vj
    from physics import CurrentClosure
    from vmex.core import implicit as im
    from vmex.core.solver import prepare_runtime
    from vmex.core.profiles import MU0

    inp = vj.VmecInput.from_file(entry.REFERENCE / "input.rotating_ellipse").change_resolution(
        mpol=2, ntor=1, ntheta=8, nzeta=8
    )
    inp = replace(pressure_input(inp, 30), ns_array=np.array([51]), curtor=-9000.0)
    closure = CurrentClosure(inp, 51, kinetic_profiles(30))
    q = np.random.default_rng(9).normal(size=closure.size) * 0.003
    q[0] = -0.09
    native = closure.input(inp, q)
    reread = vj.VmecInput.from_file(native.to_indata(tmp_path / "input.chebyshev"))
    assert reread.pcurr_type == "chebyshev_ip"
    np.testing.assert_allclose(closure.coordinates(reread), q, rtol=1e-14, atol=1e-16)
    assert reread.phiedge == inp.phiedge and reread.pres_scale == inp.pres_scale
    cfg = im.make_config(reread, device=jax.devices()[0])
    implicit = im.runtime_from_params(im.params_from_input(reread), cfg)
    ordinary = prepare_runtime(reread)
    np.testing.assert_allclose(ordinary.setup.icurv, implicit.setup.icurv, rtol=3e-14, atol=1e-17)
    integral = np.polynomial.Chebyshev(S.CURRENT_SCALE_A * q, domain=[0, 1]).integ()
    reference = ordinary.setup.signgs * MU0 / (2 * np.pi) * (integral(np.asarray(ordinary.setup.s_half)) - integral(0))
    reference[0] = 0
    np.testing.assert_allclose(ordinary.setup.icurv, reference, rtol=3e-14, atol=1e-17)
    np.testing.assert_allclose(reread.curtor, integral(1) - integral(0), rtol=1e-14)


def test_legacy_current_migration_preserves_polynomial_and_rejects_unknown_kind():
    import vmex as vj
    from physics import CurrentClosure
    from vmex.core.profiles import current

    inp = pressure_input(vj.VmecInput.from_file(entry.REFERENCE / "input.rotating_ellipse"), 30)
    ac = np.zeros_like(inp.ac)
    ac[:4] = [1.0, -0.4, 0.1, -0.03]
    inp = replace(inp, ac=ac, curtor=-9000.0)
    closure = CurrentClosure(inp, 51, kinetic_profiles(30))
    native = closure.input(inp, closure.coordinates(inp))
    s = jnp.linspace(0, 1, 101)

    def normalized(deck):
        return (
            deck.curtor
            * current(deck.pcurr_type, deck.ac, None, None, s)
            / current(deck.pcurr_type, deck.ac, None, None, 1.0)
        )

    np.testing.assert_allclose(normalized(native), normalized(inp), rtol=1e-13, atol=1e-10)
    with pytest.raises(ValueError, match="requires"):
        closure.coordinates(replace(inp, pcurr_type="line_segment_ip"))


def test_main_constraints_and_contract_are_shared_and_json_stable():
    limits, _ = entry.coil_modules()
    assert limits.P is S.P
    assert S.P.COIL_ORDER == 16 and S.P.N_SEGMENTS == 256
    assert limits.CURVATURE_POINTS == 1024 and limits.DISTANCE_POINTS == 256
    assert limits.VERIFY_POINTS == 4096
    assert (S.P.CURVATURE_LIMIT, S.P.MSC_LIMIT) == (5.0, 5.0)
    assert json.loads(json.dumps(entry.contract())) == entry.contract()


def test_cli_requires_explicit_new_output_and_prepared_bundle():
    with pytest.raises(SystemExit):
        entry.arguments("free", [])
    with pytest.raises(SystemExit):
        entry.arguments("fixed", ["--accepted-steps", "0", "--dry-run"])
    assert entry.arguments("fixed", ["--dry-run"]).dry_run
    args = entry.arguments("fixed", ["--accepted-steps", "500", "--max-trials", "5000", "--dry-run"])
    assert args.accepted_steps == 500 and args.max_trials == 5000
    with pytest.raises(SystemExit):
        entry.arguments("free", ["--accepted-steps", "500", "--max-trials", "499", "--dry-run"])


def test_long_run_trial_budget_allows_backtracking_beyond_500_trials():
    from types import SimpleNamespace
    campaign = entry.CaseRun.__new__(entry.CaseRun)
    campaign.x, campaign.y = np.zeros(1), np.zeros(1)
    campaign.cache, campaign.trials, campaign.trial_budget = {}, 500, 5000
    campaign.model = SimpleNamespace(linear=SimpleNamespace(), certify=lambda *a: {"root_residual": 0.0})
    campaign.continue_root = lambda x: x.copy()
    campaign.rows = lambda y, x: np.array([x[0]**2, .3, 1., .3])
    assert campaign._evaluate(np.ones(1))["rows"][0] == 1.0
    assert campaign.trials == 501
    campaign.trials = 5000
    with pytest.raises(RootFailure, match="trial budget"):
        campaign._evaluate(np.array([2.0]))


@pytest.mark.parametrize("arm", ["fixed", "free"])
def test_native_coupled_residual_current_and_design_jvp(monkeypatch, arm):
    """Exercise real force/Redl/NESTOR tracing on tiny grids, without qualification claims."""
    import vmex as vj
    from vmex.core import implicit as im, freeboundary_implicit as fbi
    from vmex.core.solver import _initial_state
    from essos.coils import Coils
    from types import SimpleNamespace

    monkeypatch.setattr(S, "MAX_BOUNDARY_MODE", 1)
    monkeypatch.setattr(S, "CURRENT_DEGREE", 2)
    monkeypatch.setattr(S, "REDL_N_LAMBDA", 8)
    monkeypatch.setattr(S, "REDL_SURFACES", (0.15, 0.35, 0.55, 0.75, 0.85))
    inp = vj.VmecInput.from_file(entry.REFERENCE / "input.rotating_ellipse").change_resolution(
        mpol=2, ntor=1, ntheta=8, nzeta=8
    )
    inp = pressure_input(inp, 30.0)
    inp = replace(inp, ns_array=np.array([5]), ftol_array=np.array([1e-8]), curtor=1000.0)
    cfg = im.make_config(inp, device=jax.devices()[0])
    rt = im.runtime_from_params(im.params_from_input(inp), cfg)
    seed = _initial_state(rt.setup)
    coils = Coils.from_json(str(entry.REFERENCE / "coils.initial.scalar.json"))
    # A fake ordinary-stage result supplies only a tiny structural-mask seed.
    # All differentiated residual kernels below, including NESTOR, are real.
    if arm == "free":
        monkeypatch.setattr(
            fbi,
            "_solve_free_boundary_stage",
            lambda *a, **kw: SimpleNamespace(
                result=SimpleNamespace(state=seed, fsqr=0.0, fsqz=0.0, fsql=0.0, converged=True),
                rcon0=rt.rcon0,
                zcon0=rt.zcon0,
                vacuum=SimpleNamespace(bsqvac=jnp.zeros((rt.resolution.ntheta3, rt.resolution.nzeta))),
            ),
        )
    model = EquilibriumCurrentRoot(inp, coils, seed, kinetic_profiles(30.0), arm)
    assert model.linear.options.saved_linearization
    assert model.linear.options.saved_krylov_actions
    assert model.linear.options.batch == 32
    assert model.linear.options.newton_gate == 1e-5
    from device_root import DeviceRoot
    assert isinstance(model.linear, DeviceRoot)
    y = jnp.asarray(model.initial_y)
    x = jnp.asarray(model.x0)
    direction = jnp.zeros_like(y).at[-1].set(0.001)
    value, tangent = jax.jvp(model.residual, (y, x), (direction, jnp.zeros_like(x)))
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(tangent))
    h = 1e-4
    fd = (model.residual(y + h * direction, x) - model.residual(y - h * direction, x)) / (2 * h)
    np.testing.assert_allclose(tangent, fd, rtol=3e-4, atol=1e-7)

    design_direction = jnp.zeros_like(x).at[0].set(0.01)
    _, design_tangent = jax.jvp(model.residual, (y, x), (jnp.zeros_like(y), design_direction))
    design_fd = (model.residual(y, x + h * design_direction) - model.residual(y, x - h * design_direction)) / (2 * h)
    np.testing.assert_allclose(design_tangent, design_fd, rtol=3e-4, atol=1e-7)

    from linear_root import _tangent_apply, _tangent_transpose

    cache_sizes = []
    for offset in (0.0, 1e-4, -1e-4):
        state, point = y + offset * direction, x + offset * design_direction
        saved = model.linear._prepare_tangent(state, point)
        for action, actual in [(_tangent_apply, model.linear._jvp), (_tangent_transpose, model.linear._vjp)]:
            got = np.asarray(action(saved, direction))
            expected = np.asarray(actual(state, point, direction))
            np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
        cache_sizes.append((_tangent_apply._cache_size(), _tangent_transpose._cache_size()))
    assert len(set(cache_sizes)) == 1, "changed physical roots must reuse the compiled saved actions"

@pytest.mark.parametrize("arm", ["fixed", "free"])
def test_failed_trial_keeps_last_accepted_equilibrium_and_current(arm):
    from types import SimpleNamespace

    campaign = entry.CaseRun.__new__(entry.CaseRun)
    campaign.arm = arm
    campaign.x = np.array([0.0])
    campaign.y = np.array([2.0, 3.0])
    campaign.trials = 0
    campaign.cache = {}
    campaign.event = lambda **kw: None
    seeds = []

    def fail(seed, point):
        seeds.append(seed.copy())
        seed[:] = 99.0  # even a misbehaving corrector cannot mutate accepted state
        raise RootFailure("injected failed corrector")

    campaign.model = SimpleNamespace(
        correct_trial=fail,
        linear=SimpleNamespace(),
        seed_at=lambda y, *a: y.copy(),
    )
    with pytest.raises(RootFailure, match="bounded continuation"):
        campaign.evaluate(np.array([1.0]))
    np.testing.assert_array_equal(campaign.x, [0.0])
    np.testing.assert_array_equal(campaign.y, [2.0, 3.0])
    assert len(seeds) == 3 and not campaign.cache
    for seed in seeds:
        np.testing.assert_array_equal(seed, [2.0, 3.0])


def test_singular_coupled_root_has_recoverable_failure_type():
    solver = LinearRoot(lambda y, x: jnp.array([y[0], 0.0]), options=Options(max_dofs=4))
    with pytest.raises(RootFailure, match="factorization"):
        solver.rebuild(np.array([0.0, 0.0]), np.array([0.0]))
    assert solver.factors is None


@pytest.mark.parametrize("fail_all", [False, True])
@pytest.mark.parametrize("arm", ["fixed", "free"])
@pytest.mark.parametrize("failure_phase", ["correct", "certify"])
def test_continuation_retains_certified_prefix_without_acceptance(fail_all, arm, failure_phase):
    from types import SimpleNamespace
    from device_root import DeviceRoot

    campaign = entry.CaseRun.__new__(entry.CaseRun)
    campaign.arm = arm
    campaign.x, campaign.y = np.array([0.0]), np.array([0.0, 3.0])
    seeds, certified, events = [], [], []
    campaign.event = lambda **kw: events.append(kw)
    engine = DeviceRoot(lambda y, x: y, options=Options(saved_linearization=True, trial_dense_rebuilds=1))
    engine.factors = (np.array([[0.]]), np.array([0], dtype=np.int32))
    engine._factor_point = (campaign.y.copy(), campaign.x.copy())
    engine.initialize_accepted(campaign.y, campaign.x, remaining=10)
    accepted_factors = engine.factors
    engine.last_solve = {"iterations": 6, "minimum_alpha": 1.0}
    factors_before, should_fail = [], [False]

    def correct(seed, point):
        seeds.append((seed.copy(), point.copy()))
        factors_before.append(engine.factors[0][0, 0])
        engine.factors = (np.array([[point[0]]]), np.array([0], dtype=np.int32))
        engine._factor_point = (seed.copy(), point.copy())
        # Reject the full jump and then subdivide only the second half.
        should_fail[0] = fail_all or abs(point[0] - 1.0) < 1e-12 and seed[0] <= 0.5
        if should_fail[0] and failure_phase == "correct":
            seed[:] = 99  # failed candidates must not mutate a certified prefix
            raise SlowProgress("injected difficult segment")
        return np.array([point[0], 3.0])

    def certify(y, point):
        if should_fail[0] and failure_phase == "certify":
            raise RootFailure("injected failed physical certificate")
        np.testing.assert_array_equal(y, [point[0], 3.0])
        certified.append(float(point[0]))
        return {"root_residual": 1e-14}

    campaign.model = SimpleNamespace(
        seed_at=lambda y, *args: y.copy(),
        correct_trial=correct,
        certify=certify,
        linear=engine,
    )
    if fail_all:
        with pytest.raises(RootFailure, match="bounded continuation"):
            campaign.continue_root(np.array([1.0]))
        assert len(seeds) == 3 and not certified
        assert factors_before == [0., 0., 0.]
        assert engine.factors is accepted_factors
    else:
        np.testing.assert_array_equal(campaign.continue_root(np.array([1.0])), [1.0, 3.0])
        assert certified[:2] == [0.5, 0.75]
        # The retry starts from the certified half point, not the failed root.
        np.testing.assert_array_equal(seeds[3][0], [0.5, 3.0])
        assert certified[-1] == 1.0
        assert factors_before == [0., 0., .5, .5, .75]
        assert engine.factors[0][0, 0] == 1.
    assert engine._accepted[0] is accepted_factors
    assert engine._accepted_key == campaign.x.tobytes()
    np.testing.assert_array_equal(campaign.x, [0.0])
    np.testing.assert_array_equal(campaign.y, [0.0, 3.0])
    assert any(e["event"] == "continuation_retry" for e in events)


def test_accept_discards_trials_from_previous_equilibrium_lineage():
    from types import SimpleNamespace
    campaign = entry.CaseRun.__new__(entry.CaseRun)
    campaign.model = SimpleNamespace(linear=SimpleNamespace())
    campaign.x = np.array([0.0])
    campaign.y = np.array([2.0, 3.0])
    campaign.steps = 0
    point = np.array([1.0])
    root = np.array([4.0, 5.0])
    campaign.cache = {
        point.tobytes(): dict(x=point, y=root),
        np.array([2.0]).tobytes(): dict(x=np.array([2.0]), y=np.array([6.0, 7.0])),
    }
    campaign.record = lambda: None
    campaign.accept(point)
    assert list(campaign.cache) == [point.tobytes()] and campaign.steps == 1
    point[:] = 99.0
    root[:] = 99.0
    np.testing.assert_array_equal(campaign.x, [1.0])
    np.testing.assert_array_equal(campaign.y, [4.0, 5.0])


@pytest.mark.parametrize("invalid_diagnostic", [False, True])
def test_independent_verification_uses_supplied_state_and_rejects_nan(monkeypatch, invalid_diagnostic):
    from types import SimpleNamespace
    import physics
    import vmex as vj

    inp = vj.VmecInput.from_file(entry.REFERENCE / "input.rotating_ellipse")
    inp = replace(pressure_input(inp, 30.0), ns_array=np.array([5]), curtor=1000.0)
    seed, solved, runtime = object(), object(), object()
    calls = []

    def solve(deck, **kwargs):
        calls.append(kwargs["initial_state"])
        return SimpleNamespace(state=solved, runtime=runtime)

    closure = SimpleNamespace(
        input=lambda inp, q: inp,
        coordinates=lambda deck: np.array([1.0]),
        params=lambda base, q: base,
        diagnostics=lambda state, rt, q: dict(
            redl_closure_relative=0.0,
            redl_interior_relative=np.nan if invalid_diagnostic else 0.0,
            plasma_current_A=1000.0,
        ),
    )
    monkeypatch.setattr(physics, "CurrentClosure", lambda *args: closure)
    monkeypatch.setattr(physics.opt, "solve_equilibrium", solve)
    monkeypatch.setattr(physics.opt.CoilParameters, "from_coils", lambda *args, **kwargs: SimpleChart())
    monkeypatch.setattr(
        physics,
        "evaluate_forces",
        lambda *args: (
            None,
            SimpleNamespace(fsqr=0.0, fsqz=0.0, fsql=0.0),
            SimpleNamespace(jacobian_sign_changed=False),
        ),
    )

    class SimpleChart:
        x0 = np.array([0.0])

        def __call__(self, x):
            return None

    if invalid_diagnostic:
        with pytest.raises(RootFailure, match="nonfinite NS201 current certificate"):
            physics.verify_high_resolution(inp, None, None, "fixed", initial_state=seed)
    else:
        _, state, rt, certificate = physics.verify_high_resolution(inp, None, None, "fixed", initial_state=seed)
        assert state is solved and rt is runtime
        assert certificate["coupled_newton_root_certified"] is False
    assert calls == [seed]


@pytest.mark.parametrize("fail_second", [False, True])
def test_bootstrap_preparation_preserves_warm_state_and_never_cold_retries(monkeypatch, fail_second):
    from types import SimpleNamespace
    import physics
    import vmex as vj

    inp = pressure_input(vj.VmecInput.from_file(entry.REFERENCE / "input.rotating_ellipse"), 20.0)
    closure = SimpleNamespace(
        input=lambda inp, q: inp,
        size=1,
        vander=np.ones((1, 1)),
        params=lambda base, q: replace(base, curtor=1000.0 * q[0]),
        target=lambda state, rt: np.array([1.0]),
    )
    monkeypatch.setattr(physics, "CurrentClosure", lambda *args: closure)
    monkeypatch.setattr(S, "PICARD_RTOL", 0.01)
    initial = object()
    states = []
    calls = []

    def solve(deck, **kwargs):
        calls.append(kwargs["initial_state"])
        if fail_second and len(calls) == 2:
            raise RootFailure("injected warm solve failure")
        state = object()
        states.append(state)
        return SimpleNamespace(state=state, runtime=None, result=SimpleNamespace(fsqr=1e-12, fsqz=1e-12, fsql=1e-12))

    monkeypatch.setattr(physics.opt, "solve_equilibrium", solve)
    if fail_second:
        with pytest.raises(RootFailure, match="injected warm solve failure"):
            physics.bootstrap_seed(inp, None, initial_state=initial, device="cpu")
        assert len(calls) == 2
    else:
        result = physics.bootstrap_seed(inp, None, initial_state=initial, device="cpu")
        assert result.converged and result.history[-1]["delta"] <= 0.01
        assert 990 <= result.input.curtor <= 1000
    assert calls[0] is initial
    for index in range(1, len(calls)):
        assert calls[index] is states[index - 1]


@pytest.mark.parametrize("transpose", [False, True])
def test_dense_recovery_refines_against_actual_tangent(monkeypatch, transpose):
    # Emulate a dense approximation differing from the checked tangent.
    from scipy.linalg import lu_factor

    matrix = np.array([[2.0, 0.3], [0.1, 1.0]])
    events = []
    solver = LinearRoot(
        lambda y, x: jnp.asarray(matrix) @ y - x, options=Options(max_dofs=4), event=lambda **kw: events.append(kw)
    )

    def approximate(y, x):
        solver.factors = lu_factor(matrix + 1e-4 * np.eye(2))

    monkeypatch.setattr(solver, "rebuild", approximate)
    rhs = np.array([0.7, -0.2])
    result = solver.linear(np.zeros(2), np.zeros(2), rhs, transpose=transpose)
    np.testing.assert_allclose((matrix.T if transpose else matrix) @ result, rhs, rtol=1e-9, atol=1e-12)
    assert any(e["event"] == "linear_refinement" for e in events)


def test_shared_optimizer_has_identical_rejections_and_accepted_iterates():
    from types import SimpleNamespace
    from optimizer_driver import optimize

    traces = []
    for arm in ("fixed", "free"):
        saved, rejected, events = [], [], []
        campaign = SimpleNamespace(
            x=np.array([0.0]),
            arm=arm,
            steps=5,
            trials=0,
            args=SimpleNamespace(accepted_steps=2),
            event=lambda **kw: events.append(kw),
            coil_rows=lambda x: np.ones(1),
            coil_jac=lambda x: np.zeros((1, 1)),
        )

        def evaluate(x):
            if abs(x[0]) > 0.2:
                rejected.append(float(x[0]))
                raise RootFailure("trial geometry outside local basin")
            return dict(rows=np.array([10 * (x[0] - 0.05) ** 2, 0.3, 1.0, 0.3]))

        def accept(x):
            evaluate(x)
            campaign.x = x.copy()
            campaign.steps += 1
            saved.append(float(x[0]))

        campaign.evaluate = evaluate
        campaign.jacobian = lambda x: np.array([[20 * (x[0] - 0.05)], [0.0], [0.0], [0.0]])
        campaign.accept = accept
        result = optimize(campaign)
        assert rejected and saved and all(abs(x) <= 0.2 for x in saved)
        assert result["new_accepted_steps"] <= 2 and result["accepted_steps"] == 5 + len(saved)
        assert abs(campaign.x[0] - 0.05) < 1e-8
        assert any(e["event"] == "optimizer_trial_rejected" for e in events)
        traces.append((saved, rejected, events, result))
    assert traces[0] == traces[1]


def test_prepared_migration_never_allows_physics_or_core_changes():
    import copy

    saved = copy.deepcopy(entry.contract())
    saved["physics"].pop("BOUNDARY_SPECTRAL_ALPHA")
    saved["physics"]["BOUNDARY_MAX_DISPLACEMENT_M"] = 0.001
    saved["physics"]["CONTINUATION_INITIAL_FREE"] = saved["physics"].pop("CONTINUATION_INITIAL")
    saved["numerical_sources"].pop("finite-beta-bootstrap-benchmark/optimizer_driver.py")
    saved["numerical_sources"]["finite-beta-bootstrap-benchmark/physics.py"] = "old-version"
    with pytest.raises(ValueError, match="contract"):
        entry.validate_prepared_contract(saved)
    assert entry.validate_prepared_contract(saved, allow_code_update=True)
    for section, key in [
        ("physics", "DENSITY_AXIS_M3"),
        ("shared_main", "CURVATURE_LIMIT"),
        ("core_sources", "solver.py"),
    ]:
        invalid = copy.deepcopy(saved)
        invalid[section][key] = "changed"
        with pytest.raises(ValueError):
            entry.validate_prepared_contract(invalid, allow_code_update=True)


@pytest.mark.parametrize("arm", ["fixed", "free"])
def test_requested_iota_floor_reaches_actual_optimizer_without_changing_vacuum(monkeypatch, arm):
    from types import SimpleNamespace
    from optimizer_driver import build_stage, run_optimizer
    from main_reference import vacuum_script
    from vmex import optimize as opt

    reference = vacuum_script(arm)
    original_floor = reference.IOTA_FLOOR
    campaign = SimpleNamespace(
        x=np.array([0.0]),
        steps=0,
        args=SimpleNamespace(accepted_steps=100),
        event=lambda **kw: None,
        accept=lambda x: None,
        evaluate=lambda x: {"rows": np.array([0.0, x[0], 1.0, 0.3])},
        jacobian=lambda x: np.array([[0.0], [1.0], [0.0], [0.0]]),
        coil_rows=lambda x: np.ones(1),
        coil_jac=lambda x: np.zeros((1, 1)),
    )
    captured = []

    def minimize(problem, **kwargs):
        constraint = kwargs["constraints"][0]
        assert kwargs["method"] == "SLSQP"
        assert kwargs["options"] == dict(maxiter=100, ftol=S.P.OPTIMIZER_FTOL)
        # Check the physical boundary of the constraint actually handed to SLSQP.
        for offset in (-1e-5, 1e-5):
            value = constraint.fun(np.array([0.28 + S.P.IOTA_MARGIN + offset]))[0]
            assert (value >= np.asarray(constraint.lb)[0]) == (offset > 0)
        captured.append(True)
        raise RuntimeError("stop after inspecting actual constraints")

    monkeypatch.setattr(opt, "minimize", minimize)
    with pytest.raises(RuntimeError, match="inspecting"):
        run_optimizer(build_stage(campaign), campaign.args, arm=arm)
    assert captured and S.IOTA_FLOOR == 0.28
    assert S.P.IOTA_FLOOR == 0.19 and reference.IOTA_FLOOR == original_floor


def test_prepared_reference_can_change_iota_goal_without_changing_pressure():
    import copy

    saved = copy.deepcopy(entry.contract())
    saved["optimization_targets"] = {"iota_floor": 0.19}
    with pytest.raises(ValueError, match="contract"):
        entry.validate_prepared_contract(saved)
    assert entry.validate_prepared_contract(saved, allow_code_update=True) == []
    saved["physics"]["BETA_AXIS_INITIAL"] = 0.05
    with pytest.raises(ValueError, match="physics"):
        entry.validate_prepared_contract(saved, allow_code_update=True)


def test_prepared_chebyshev_migration_allows_only_the_pinned_profile_update():
    import copy

    saved = copy.deepcopy(entry.contract())
    saved["core_sources"]["profiles.py"] = "7cbc5b1e62d73163e51e70ee3f43a402db1803608d51e2b5cd30bb93558688a5"
    with pytest.raises(ValueError):
        entry.validate_prepared_contract(saved)
    assert entry.validate_prepared_contract(saved, allow_code_update=True) == ["vmex/core/profiles.py"]
    saved["core_sources"]["profiles.py"] = "unknown-profile-implementation"
    with pytest.raises(ValueError, match="core"):
        entry.validate_prepared_contract(saved, allow_code_update=True)


def test_boundary_seed_transport_preserves_axis_current_and_imposed_edge(monkeypatch):
    import vmex as vj
    from vmex.core.solver import _initial_state
    from essos.coils import Coils

    monkeypatch.setattr(S, "MAX_BOUNDARY_MODE", 3)
    inp = vj.VmecInput.from_file(entry.REFERENCE / "input.rotating_ellipse").change_resolution(
        mpol=4, ntor=2, ntheta=12, nzeta=12
    )
    inp = replace(pressure_input(inp, 20.0), ns_array=np.array([9]), curtor=1000.0)
    from vmex.core import implicit as im
    from vmex import optimize as opt

    cfg = im.make_config(inp, device=jax.devices()[0])
    rt = im.runtime_from_params(im.params_from_input(inp), cfg)
    state = _initial_state(rt.setup)
    coils = Coils.from_json(str(entry.REFERENCE / "coils.initial.scalar.json"))
    model = EquilibriumCurrentRoot(inp, coils, state, kinetic_profiles(20.0), "fixed")
    x0, y = model.x0, model.initial_y
    k = opt._dof_modes(inp, S.MAX_BOUNDARY_MODE).index((1, 0))
    trial = x0.copy()
    # A 2 mm edge change exceeds the removed fixed-only cap. Transport it;
    # the subsequent nonlinear solve/certification decides whether it is usable.
    trial[k] = 0.002 / model.scales[k]
    transported = model.transport_seed(y, x0, trial)
    np.testing.assert_array_equal(transported[model.nstate :], y[model.nstate :])
    before, _, _, _ = model.objects(jnp.asarray(y), jnp.asarray(x0))
    raw, _, _, _ = model.objects(jnp.asarray(y), jnp.asarray(trial))
    after, _, _, _ = model.objects(jnp.asarray(transported), jnp.asarray(trial))
    for name in before.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(after, name)[0], getattr(before, name)[0])
        np.testing.assert_array_equal(getattr(after, name)[-1], getattr(raw, name)[-1])
    assert np.max(np.abs(np.asarray(after.R_cos)[1:-1] - np.asarray(before.R_cos)[1:-1])) > 0
    np.testing.assert_array_equal(y, model.initial_y)


def test_full_root_predictor_includes_current_and_requires_nonlinear_correction():
    def residual(y, x):
        z, q = y
        return jnp.array([z + 0.2 * z * z - q - x[0], q - 0.3 * z * z - 0.4 * x[0]])

    solver = LinearRoot(residual, options=Options(max_dofs=4, atol=1e-13))
    x = np.array([0.2])
    root = solver.solve(np.array([0.1, 0.1]), x)
    saved = root.copy()
    trial = x + 1e-3
    predicted = solver.predict(root, x, trial)
    assert predicted[1] != root[1]
    error = float(jnp.linalg.norm(residual(predicted, trial)))
    assert 1e-13 < error < 0.01 * float(jnp.linalg.norm(residual(root, trial)))
    corrected = solver.solve(predicted, trial)
    assert float(jnp.linalg.norm(residual(corrected, trial))) <= 1e-13
    np.testing.assert_array_equal(root, saved)


@pytest.mark.parametrize("arm", ["fixed", "free"])
def test_trial_seed_chooses_valid_lowest_residual_without_accepting_prediction(arm):
    from types import SimpleNamespace

    model = EquilibriumCurrentRoot.__new__(EquilibriumCurrentRoot)
    model.arm = arm
    model.transport_seed = lambda *a: np.array([1.5])
    model.linear = SimpleNamespace(predict=lambda *a: np.array([2.0]), residual=lambda y, x: y - x)
    model.valid = lambda y, x: bool(np.all(y > 0))
    events = []
    model.event = lambda **kw: events.append(kw)
    accepted = np.array([1.0])
    np.testing.assert_array_equal(model.seed_at(accepted, accepted, np.array([2.0])), [2.0])
    np.testing.assert_array_equal(accepted, [1.0])
    assert events[-1]["method"] == "accepted_tangent"
    model.linear.predict = lambda *a: np.array([-2.0])
    np.testing.assert_array_equal(model.seed_at(accepted, accepted, np.array([2.0])), [1.5 if arm == "fixed" else 1.0])


def test_reference_import_restores_unrelated_parameters_and_search_path():
    from types import SimpleNamespace
    from main_reference import vacuum_script, P

    previous, paths = sys.modules.get("parameters"), sys.path.copy()
    sentinel = SimpleNamespace(unrelated=True)
    sys.modules["parameters"] = sentinel
    vacuum_script.cache_clear()
    try:
        for arm in ("fixed", "free"):
            reference = vacuum_script(arm)
            assert reference.P is P
            assert sys.modules["parameters"] is sentinel and sys.path == paths
    finally:
        if previous is None:
            sys.modules.pop("parameters", None)
        else:
            sys.modules["parameters"] = previous


def test_free_damped_predictor_selects_lowest_valid_coupled_residual():
    from types import SimpleNamespace

    model = EquilibriumCurrentRoot.__new__(EquilibriumCurrentRoot)
    model.arm = "free"
    model.valid = lambda y, x: bool(np.all(y > 0))
    model.event = lambda **kw: None
    model.linear = SimpleNamespace(predict=lambda *a: np.array([5.0]), residual=lambda y, x: y - 2)
    accepted = np.array([1.0])
    np.testing.assert_array_equal(model.seed_at(accepted, np.array([0.0]), np.array([1.0])), [2.0])
    np.testing.assert_array_equal(accepted, [1.0])


@pytest.mark.parametrize("mode", ["success", "slow", "hard_failure"])
@pytest.mark.parametrize("arm", ["fixed", "free"])
def test_coupled_newton_precedes_optional_ordinary_recovery(mode, arm):
    model = EquilibriumCurrentRoot.__new__(EquilibriumCurrentRoot)
    model.arm = arm
    model.event = lambda **kw: None
    model.valid = lambda *a: True
    solver = LinearRoot(
        lambda y, x: jnp.array([y[0] - y[1] - x[0], y[1] - 0.3 * y[0] ** 2]), options=Options(atol=1e-13, max_dofs=4)
    )
    model.linear = solver
    original, calls, ordinary = solver.solve, [], []

    def solve(y, x):
        calls.append(y.copy())
        if len(calls) == 1 and mode != "success":
            y[:] = 99  # failed correction cannot corrupt the recovery seed
            raise (SlowProgress if mode == "slow" else RootFailure)("injected recovery")
        return original(y, x)

    def forward(y, x):
        ordinary.append(y.copy())
        return np.array([0.21, 0.01])

    solver.solve, model.forward_seed = solve, forward
    seed, point = np.array([0.1, 0.01]), np.array([0.2])
    if mode == "slow":
        with pytest.raises(SlowProgress):
            model.correct_trial(seed, point)
    else:
        root = model.correct_trial(seed, point)
        assert np.linalg.norm(solver.residual(root, point)) <= 1e-13
        assert root[1] != seed[1]
    assert len(ordinary) == (1 if mode == "hard_failure" else 0)
    assert len(calls) == (2 if mode == "hard_failure" else 1)
    np.testing.assert_array_equal(seed, [0.1, 0.01])


@pytest.mark.parametrize("mode", ["improved", "worse", "invalid", "failed"])
def test_ordinary_forward_seed_never_bypasses_coupled_current_polish(mode):
    # The ordinary solve may fix force balance at predicted current, but it
    # cannot certify current closure or be directly accepted by the optimizer.
    from vmex.core.errors import VmecError

    solver = LinearRoot(
        lambda y, x: jnp.array([y[0] - y[1] - x[0], y[1] - 0.3 * y[0] ** 2]), options=Options(atol=1e-13, max_dofs=4)
    )
    model = EquilibriumCurrentRoot.__new__(EquilibriumCurrentRoot)
    model.arm = "fixed"
    model.linear = solver
    model.valid = lambda y, x: bool(np.all(y >= 0))
    events = []
    model.event = lambda **kw: events.append(kw)
    seed = np.array([0.1, 0.01])
    supplied = []

    def forward(y, x):
        supplied.append(y.copy())
        if mode == "failed":
            raise VmecError("ordinary convergence failed")
        return {"improved": np.array([0.21, 0.01]), "worse": np.array([0.9, 0.01]), "invalid": np.array([-1.0, 0.01])}[
            mode
        ]

    model.forward_seed = forward
    forward = model.ordinary_seed(seed.copy(), np.array([0.2]))
    result = model.linear.solve(forward, np.array([0.2]))
    assert np.linalg.norm(solver.residual(result, np.array([0.2]))) <= 1e-13
    assert result[1] != seed[1]  # current was corrected, not frozen
    np.testing.assert_array_equal(seed, [0.1, 0.01])
    np.testing.assert_array_equal(supplied[0], seed)
    if mode in ("improved", "worse"):
        assert events[0]["selected"] == (mode == "improved")
    else:
        assert events[0]["event"] == "ordinary_correction_rejected"


@pytest.mark.parametrize("arm", ["fixed", "free"])
def test_native_forward_adapter_preserves_chart_and_uses_warm_state(monkeypatch, arm):
    import vmex as vj
    import physics
    from vmex.core import implicit as im, freeboundary_implicit as fbi
    from vmex.core.solver import _initial_state
    from essos.coils import Coils
    from types import SimpleNamespace

    monkeypatch.setattr(S, "MAX_BOUNDARY_MODE", 1)
    monkeypatch.setattr(S, "CURRENT_DEGREE", 2)
    inp = vj.VmecInput.from_file(entry.REFERENCE / "input.rotating_ellipse").change_resolution(
        mpol=2, ntor=1, ntheta=8, nzeta=8
    )
    inp = replace(pressure_input(inp, 30.0), ns_array=np.array([5]), curtor=1000.0)
    cfg = im.make_config(inp, device=jax.devices()[0])
    rt = im.runtime_from_params(im.params_from_input(inp), cfg)
    initial = _initial_state(rt.setup)
    coils = Coils.from_json(str(entry.REFERENCE / "coils.initial.scalar.json"))
    # Mock only the expensive forward integration; use the actual spectral
    # chart, deck construction and current conversion on both sides.
    monkeypatch.setattr(
        fbi,
        "_solve_free_boundary_stage",
        lambda *a, **kw: SimpleNamespace(
            result=SimpleNamespace(state=initial, fsqr=0.0, fsqz=0.0, fsql=0.0, converged=True),
            rcon0=rt.rcon0,
            zcon0=rt.zcon0,
            vacuum=SimpleNamespace(bsqvac=jnp.zeros((rt.resolution.ntheta3, rt.resolution.nzeta))),
        ),
    )
    model = EquilibriumCurrentRoot(inp, coils, initial, kinetic_profiles(30.0), arm)
    y, x = model.initial_y.copy(), model.x0.copy()
    y[-1] += 1e-6
    expected, _, params, _ = model.objects(jnp.asarray(y), jnp.asarray(x))
    calls = []

    def ordinary(deck, **kw):
        calls.append(kw)
        assert float(deck.curtor) == pytest.approx(float(params.curtor))
        assert float(deck.phiedge) == float(inp.phiedge)
        for name in expected.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(kw["initial_state"], name), getattr(expected, name))
        return SimpleNamespace(state=expected, result=SimpleNamespace(state=expected))

    monkeypatch.setattr(physics.opt, "solve_equilibrium", ordinary)
    monkeypatch.setattr(fbi, "_solve_free_boundary_stage", ordinary)
    result = model.forward_seed(y, x)
    np.testing.assert_allclose(result, y, rtol=0, atol=1e-14)
    assert len(calls) == 1
    if arm == "free":
        assert calls[0]["jacobian_retries"] == 0 and not calls[0]["allow_initial_axis_reguess"]
        assert calls[0]["constraint_continuation"] == (model.rcon, model.zcon)
    else:
        assert calls[0]["raise_on_max_iterations"] and not calls[0]["polish_force_balance"]
        assert calls[0]["jacobian_retries"] == 0 and not calls[0]["coarse_grid_retry"]


def test_production_cli_excludes_gradient_experiments():
    for arm in ("fixed", "free"):
        with pytest.raises(SystemExit):
            entry.arguments(arm, ["--dry-run", "--verify-gradients"])


def test_fresh_coil_selection_rejects_infeasible_and_nonfinite_candidates():
    from fit_initial_coils import choose_feasible, spectral_scales

    rows = [
        dict(objective=0.0, minimum_scaled_slack=-1.0),
        dict(objective=float("nan"), minimum_scaled_slack=1.0),
        dict(objective=2.0, minimum_scaled_slack=0.0),
        dict(objective=3.0, minimum_scaled_slack=1.0),
    ]
    assert choose_feasible(rows) is rows[2]
    with pytest.raises(RuntimeError, match="no sampled-feasible"):
        choose_feasible(rows[:2])
    scales = spectral_scales(16, 3, 0.05)
    assert scales.shape == (297,) and np.all(scales > 0)
    assert scales[0] > scales[-1]


def test_fixed_coil_fit_field_matches_public_virtual_casing_interface():
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart
    from vmex.core.virtual_casing import PlasmaVacuumInterface
    from fit_initial_coils import normal_on_fixed_interface

    phi, theta = jnp.meshgrid(jnp.arange(3) / 3 * jnp.pi, jnp.arange(4) / 4 * 2 * jnp.pi, indexing="ij")
    gamma = jnp.stack(
        ((1 + 0.2 * jnp.cos(theta)) * jnp.cos(phi), (1 + 0.2 * jnp.cos(theta)) * jnp.sin(phi), 0.2 * jnp.sin(theta))
    )
    normal = jnp.stack((jnp.cos(theta) * jnp.cos(phi), jnp.cos(theta) * jnp.sin(phi), jnp.sin(theta)))
    plasma = jnp.full_like(gamma, 0.003)
    vc = PlasmaVacuumInterface(
        gamma,
        normal,
        jnp.ones((3, 4)) / 12,
        phi[:, 0],
        jnp.sum(plasma * normal, axis=0),
        plasma,
        jnp.ones((3, 4)),
        0.0,
        2,
    )
    coils = Coils(
        CreateEquallySpacedCurves(3, 2, 1.0, 0.6, n_segments=24, nfp=2, stellsym=True), jnp.array([2.7e5] * 3)
    )
    field = BiotSavart(coils)

    def external(xyz):
        return jax.vmap(field.B)(xyz.reshape(-1, 3)).reshape(xyz.shape)

    expected = vc.bnormal_residual(external) / jnp.linalg.norm(vc.total_B_out(external), axis=0)
    np.testing.assert_allclose(normal_on_fixed_interface(coils, vc), expected, rtol=1e-13, atol=1e-14)


def test_coil_feasibility_recovery_never_exports_invalid_slsqp_proposal():
    from fit_initial_coils import feasible_segment

    anchor = np.array([0.0, 0.0])
    trial = np.array([2.0, 1.0])

    def constraints(x):
        return np.array([1.0 - np.linalg.norm(x), x[0] + 0.1])

    recovered = feasible_segment(anchor, trial, constraints)
    assert np.min(constraints(recovered)) >= -1e-8
    assert 0.99 < np.linalg.norm(recovered) <= 1.00000001
    np.testing.assert_array_equal(anchor, [0.0, 0.0])
    np.testing.assert_array_equal(trial, [2.0, 1.0])
    with pytest.raises(RuntimeError, match="valid anchor"):
        feasible_segment(trial, anchor, constraints)


@pytest.mark.usefixtures("_module_jit_enabled")
@pytest.mark.parametrize("arm", ["fixed", "free"])
def test_device_backend_through_production_slsqp_acceptance(arm):
    from types import SimpleNamespace
    from device_root import DeviceRoot
    from optimizer_driver import optimize

    residual = lambda y, x: jnp.array([2*y[0]-y[1]-x[0], y[1]-.2*y[0]])
    engine = DeviceRoot(residual, options=Options(max_dofs=4, batch=2, saved_linearization=True,
                        newton_rtol=1e-6, newton_gate=1e-5))
    campaign = entry.CaseRun.__new__(entry.CaseRun)
    campaign.x, campaign.y = np.array([.4]), np.array([.4/1.8, .08/1.8])
    campaign.arm, campaign.steps, campaign.trials = arm, 0, 0
    campaign.cache, campaign.rejected = {}, {}
    campaign.args = SimpleNamespace(accepted_steps=2)
    events, accepted = [], []
    campaign.event = lambda **kw: events.append(kw)
    campaign.rows = jax.jit(lambda y, x: jnp.array([.5*(y[0]-.5)**2, y[0], 1., .3]))
    campaign.coil_rows = lambda x: np.ones(1)
    campaign.coil_jac = lambda x: np.zeros((1, 1))

    def certify(y, x):
        norm = np.linalg.norm(residual(y, x))
        assert norm <= 1e-12
        return dict(root_residual=float(norm))

    campaign.model = SimpleNamespace(linear=engine,
        seed_at=engine.predict, correct_trial=engine.solve, certify=certify)

    def record():
        certify(campaign.y, campaign.x)
        assert engine._accepted_key == campaign.x.tobytes()
        accepted.append(campaign.x.copy())

    campaign.record = record
    result = optimize(campaign)
    assert accepted and 1 <= result['new_accepted_steps'] <= 2
    assert campaign.rows(campaign.y, campaign.x)[0] < .5*(.4/1.8-.5)**2
    assert engine._accepted_key == campaign.x.tobytes()
    assert not engine._trial


@pytest.mark.parametrize("problem", [None, "slower", "marginal", "worse_loss", "worse_slack", "gradient", "fd", "lineage"])
def test_recovery_benchmark_requires_speed_accuracy_and_comparable_progress(tmp_path, problem):
    import importlib.util

    path = CASE.parents[1] / "benchmarks/finite_beta_recovery_comparison.py"
    spec = importlib.util.spec_from_file_location("recovery_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for arm in ("fixed", "free"):
        for label in ("baseline", "candidate"):
            out = tmp_path / f"{arm}-{label}"
            out.mkdir()
            candidate = label == "candidate"
            contract = {k: {} for k in ("schema", "optimization_targets", "shared_main", "core_sources", "physics")}
            seconds = 100 if not candidate else 50
            if candidate and problem in ("slower", "marginal"):
                seconds = 150 if problem == "slower" else 95
            data = {
                "result.json": dict(status="pilot_and_fd_passed", accepted_steps=2, seconds_to_accepted_budget=seconds),
                "gradient_verification.json": dict(passed=not (candidate and problem == "fd")),
                "contract.json": contract,
                "restart.json": dict(checkpoint_sha256="wrong" if candidate and problem == "lineage" else "same",
                                     prepared_manifest_sha256="same"),
                "runtime.json": dict(versions={"jax": "same"}, hardware="same"),
                "optimization.json": dict(minimum_optimizer_slack=-.5 if candidate and problem == "worse_slack" else -.1),
            }
            for name, value in data.items():
                (out / name).write_text(json.dumps(value))
            objective = 2 if candidate and problem == "worse_loss" else 1
            (out / "accepted_steps.jsonl").write_text("".join(
                json.dumps(dict(step=step, objective=objective)) + "\n" for step in range(3)))
            gradient = np.array([[2. if candidate and problem == "gradient" else 1.]])
            np.savez(out / "initial-derivative.npz", x=np.zeros(1), rows=np.ones(1), gradient=gradient)
    if problem in ("gradient", "fd", "lineage"):
        with pytest.raises(AssertionError):
            module.compare(tmp_path)
    else:
        result = module.compare(tmp_path)
        assert result["eligible_to_replace"] == (problem is None)


@pytest.mark.parametrize('forward_converged', [False, True])
def test_free_verification_preserves_handoff_baselines_and_gates_picard(monkeypatch, forward_converged):
    """A rejected forward state never feeds Redl; an accepted one owns its baselines."""
    from dataclasses import make_dataclass
    from types import SimpleNamespace
    import physics
    import vmex as vj

    inp = replace(vj.VmecInput.from_file(entry.REFERENCE / 'input.rotating_ellipse'),
                  ns_array=np.array([5]))
    seed, solved = object(), object()
    baseline = (jnp.ones((5, 3, 4)), 2*jnp.ones((5, 3, 4)))
    updated = tuple(.9*x for x in baseline)
    calls, events, current_calls = [], [], []
    Runtime = make_dataclass('Runtime', ['rcon0', 'zcon0', 'lfreeb', 'jmax', 'presf_ns_scale', 'bsqvac_edge'])
    runtime = Runtime(None, None, False, 4, 0., None)

    class Chart:
        x0 = np.zeros(1)
        def __call__(self, x):
            return None

    def current_diagnostic(*args):
        current_calls.append(args)
        return dict(redl_closure_relative=1e-4 if len(current_calls)==1 else 0.,
                    redl_interior_relative=0., plasma_current_A=1000.)

    closure = SimpleNamespace(s=np.linspace(0, 1, 5), coordinates=lambda inp:np.ones(1),
        input=lambda inp,q:inp, params=lambda base,q:base, diagnostics=current_diagnostic,
        target=lambda state,rt:np.ones(1))
    def stage(*args, **kw):
        calls.append(kw)
        return SimpleNamespace(result=SimpleNamespace(converged=forward_converged,
            state=solved, iterations=12000 if not forward_converged else 3,
            fsqr=0., fsqz=0., fsql=0., fedge=0.),
            rcon0=updated[0], zcon0=updated[1],
            vacuum=SimpleNamespace(turned_on=forward_converged,vacuum_calls=2 if forward_converged else 0))

    monkeypatch.setattr(physics, 'CurrentClosure', lambda *args:closure)
    monkeypatch.setattr(physics.opt.CoilParameters, 'from_coils', lambda *args,**kw:Chart())
    monkeypatch.setattr(physics.fbi, 'make_free_boundary_config', lambda *args,**kw:
        SimpleNamespace(resolution=None,implicit=None,vacuum_program=SimpleNamespace(bsq=lambda *args:None)))
    monkeypatch.setattr(physics.fbi, '_solve_free_boundary_stage', stage)
    monkeypatch.setattr(physics.im, 'runtime_from_params', lambda *args:runtime)
    monkeypatch.setattr(physics.fbi, '_presf_ns_scale_traceable', lambda *args:0.)
    monkeypatch.setattr(physics,'evaluate_forces',lambda *args:(None,
        SimpleNamespace(fsqr=0.,fsqz=0.,fsql=0.,fedge=0.),SimpleNamespace(jacobian_sign_changed=False)))
    kwargs=dict(initial_state=seed,initial_baselines=baseline,event=lambda **kw:events.append(kw))
    if forward_converged:
        physics.verify_high_resolution(inp,None,None,'free',**kwargs)
        assert len(calls)==2 and calls[1]['initial_state'] is solved
        assert all(a is b for a,b in zip(calls[1]['constraint_continuation'],updated))
    else:
        with pytest.raises(RootFailure,match='vacuum_calls.*0'):
            physics.verify_high_resolution(inp,None,None,'free',**kwargs)
        assert len(calls)==1 and not current_calls
    assert calls[0]['initial_state'] is seed
    assert calls[0]['constraint_continuation'] is baseline
    assert calls[0]['include_edge_in_convergence'] is True
    assert calls[0]['edge_force_tolerance']==S.P.VERIFY_FTOL
    assert events[0]['event']=='verification_forward'


def _manufactured_interface(state, *, pressure_pa=0.):
    """Native interface with analytic live field, geometry, and area weights."""
    from vmex.core.virtual_casing import PlasmaVacuumInterface
    z, q = state
    t = jnp.arange(6., dtype=float).reshape(2, 3) / 6
    gamma = jnp.stack([1 + .1*z + .2*t, .1*t, .05*z*t])
    angle = .02*z*(1+t)
    normal = jnp.stack([jnp.sin(angle), jnp.zeros_like(t), jnp.cos(angle)])
    weights = jax.nn.softmax((t*(1+.2*z)).ravel()).reshape(t.shape)
    plasma = jnp.stack([.03*q*(1+t), .01*q*t, -.015*q*(1-t)])
    return PlasmaVacuumInterface(gamma=gamma, normal=normal, weights=weights,
        phi_grid=jnp.array([0., 1.]), Bn_plasma=jnp.sum(plasma*normal, axis=0),
        B_plasma=plasma, Bin_mag2=(1+.1*z+.02*q*t)**2, p_edge=pressure_pa, nfp=1)


@pytest.mark.parametrize('pressure_pa', [0., 1000.])
def test_boundary_pressure_residual_matches_native_interface_and_jvp(pressure_pa):
    from vmex.core.profiles import MU0
    state = jnp.array([.3, .2])
    point = jnp.array([.1, .25])
    def values(y, x):
        vc = _manufactured_interface(y, pressure_pa=pressure_pa)
        external = lambda xyz: jnp.stack([1+.1*x[1]+.02*xyz[..., 0],
            .03*x[0]+.01*xyz[..., 1], .02*x[1]+.005*xyz[..., 2]], axis=-1)
        return vc, external, entry.boundary_residuals(vc, external)

    vc, ext, (bn, jump, weights) = values(state, point)
    scale = jnp.sum(weights*(vc.Bin_mag2 + 2*MU0*pressure_pa))
    np.testing.assert_allclose(bn, vc.bnormal_residual(ext)/jnp.linalg.norm(vc.total_B_out(ext),axis=0), atol=1e-15)
    np.testing.assert_allclose(jump, vc.pressure_balance_residual(ext)/scale, atol=1e-15)
    objective = lambda y,x:entry.boundary_cost(*values(y,x)[2])
    for dy,dx in [(jnp.array([1.,0.]),jnp.zeros(2)), (jnp.array([0.,1.]),jnp.zeros(2)),
                  (jnp.zeros(2),jnp.array([0.,1.]))]:
        _, ad = jax.jvp(objective,(state,point),(dy,dx))
        assert abs(float(ad)) > 1e-3  # state, bootstrap field and coil dependence are live
        for h in (1e-5,5e-6):
            fd = (objective(state+h*dy,point+h*dx)-objective(state-h*dy,point-h*dx))/(2*h)
            np.testing.assert_allclose(ad,fd,rtol=2e-6,atol=1e-7)


def test_tangent_but_wrong_strength_field_fails_pressure_gate():
    from vmex.core.virtual_casing import PlasmaVacuumInterface
    one = jnp.ones((2, 3)); zero = jnp.zeros_like(one)
    vc = PlasmaVacuumInterface(gamma=jnp.stack([one,zero,zero]),normal=jnp.stack([zero,zero,one]),
        weights=one/6,phi_grid=jnp.array([0.,1.]),Bn_plasma=zero,
        B_plasma=jnp.stack([zero,zero,zero]),Bin_mag2=one,p_edge=0.,nfp=1)
    def residuals(amplitude):
        return entry.boundary_residuals(vc,lambda xyz:jnp.stack([
            amplitude*jnp.ones(xyz.shape[:-1]),jnp.zeros(xyz.shape[:-1]),jnp.zeros(xyz.shape[:-1])],axis=-1))
    good = residuals(1.); bad = residuals(.9)
    np.testing.assert_allclose(bad[0],0.,atol=0.)
    np.testing.assert_allclose(bad[1],-.19,atol=1e-15)
    assert float(entry.normal_cost(bad[0],bad[2]))==0.
    assert float(entry.boundary_cost(*bad))>0.
    assert entry.boundary_match_passed(entry.boundary_diagnostics(*good))
    assert not entry.boundary_match_passed(entry.boundary_diagnostics(*bad))
    invalid = entry.boundary_diagnostics(*good); invalid['pressure_balance_rms']=float('nan')
    assert not entry.boundary_match_passed(invalid)


@pytest.mark.usefixtures('_module_jit_enabled')
def test_pressure_objective_implicit_derivative_against_independent_corrected_endpoints():
    """Manufactured coupled state/current root; not a production VMEX qualification."""
    residual = lambda y,x:jnp.array([y[0]+.2*y[0]**2-y[1]-x[0],
                                    y[1]-.3*y[0]**2-.4*x[0]])
    def objectives(y,x):
        vc = _manufactured_interface(y)
        external = lambda xyz:jnp.stack([1+.1*x[1]+.02*xyz[...,0],
            .03*x[0]+jnp.zeros(xyz.shape[:-1]), .02*x[1]+jnp.zeros(xyz.shape[:-1])],axis=-1)
        bn,dp,w = entry.boundary_residuals(vc,external)
        return jnp.array([entry.normal_cost(bn,w),entry.pressure_balance_cost(dp,w)])
    solver=LinearRoot(residual,options=Options(max_dofs=4,atol=1e-13))
    x=np.array([.2,.25]); y=solver.solve(np.array([.2,.1]),x)
    _,jac=solver.derivative(objectives,y,x)
    for direction in np.eye(2):
        for h in (1e-5,5e-6):
            vals=[]
            for sign in (-1,1):
                point=x+sign*h*direction
                root=solver.solve(y.copy(),point)
                assert np.linalg.norm(np.asarray(residual(root,point)))<1e-13
                vals.append(np.asarray(objectives(root,point)))
            np.testing.assert_allclose(jac@direction,(vals[1]-vals[0])/(2*h),rtol=2e-6,atol=1e-7)


@pytest.mark.parametrize('arm',['fixed','free'])
def test_pressure_objective_is_wired_only_into_fixed_arm(monkeypatch,arm):
    from types import SimpleNamespace
    import essos.surfaces
    from vmex import optimize as opt
    campaign=entry.CaseRun.__new__(entry.CaseRun);campaign.arm=arm;campaign.precision=None
    campaign.model=SimpleNamespace(objects=lambda y,x:(y,None,None,x),
        chart=SimpleNamespace(coils_from_x=lambda c:c),design=lambda x:(None,x))
    limits=SimpleNamespace(SURFACE_GRID=(2,3),surface_distance=lambda *args:jnp.asarray(.3),
        coil_inequalities=lambda c:jnp.ones(1))
    monkeypatch.setattr(entry,'coil_modules',lambda:(limits,None))
    monkeypatch.setattr(opt,'QuasisymmetryRatioResidual',lambda *args:SimpleNamespace(residuals_state=lambda s,r:s))
    monkeypatch.setattr(opt,'aspect_ratio',lambda *args:jnp.asarray(5.))
    monkeypatch.setattr(opt,'min_abs_iota',lambda *args:jnp.asarray(.3))
    monkeypatch.setattr(opt,'major_radius',lambda *args:jnp.asarray(1.))
    monkeypatch.setattr(opt,'boundary_from_state',lambda *args:(None,None,None,None))
    monkeypatch.setattr(essos.surfaces,'surfacerzfourier_from_boundary',lambda *args,**kw:None)
    def boundary(inp,state,rt,coils,precision):
        if arm=='free': raise AssertionError('free optimizer must not evaluate interface penalties')
        return jnp.array([.002,.003]),jnp.array([.03,.04]),jnp.array([.4,.6])
    monkeypatch.setattr(entry,'boundary_values',boundary)
    campaign.configure_objectives(SimpleNamespace(nfp=1))
    y=jnp.array([.2,.1]);x=jnp.array([.3]);actual=campaign.rows(y,x)[0]
    expected=.5*float(jnp.vdot(y,y))
    if arm=='fixed':
        expected+=float(entry.boundary_cost(*boundary(None,None,None,None,None)))
        np.testing.assert_allclose(jnp.sum(campaign.interface_rows(y,x)),actual-.5*jnp.vdot(y,y))
    np.testing.assert_allclose(actual,expected)


def test_normal_only_prepared_inputs_require_explicit_objective_migration():
    import copy
    saved=copy.deepcopy(entry.contract())
    for key in list(saved['physics']):
        if key=='BOUNDARY_OBJECTIVE_VERSION' or key.startswith('PRESSURE_BALANCE_'):
            saved['physics'].pop(key)
    with pytest.raises(ValueError,match='contract'):
        entry.validate_prepared_contract(saved)
    assert entry.validate_prepared_contract(saved,allow_code_update=True)==[]
    saved['physics']['DENSITY_AXIS_M3']*=2
    with pytest.raises(ValueError,match='physics'):
        entry.validate_prepared_contract(saved,allow_code_update=True)


@pytest.mark.parametrize('arm', ['fixed', 'free'])
def test_separate_gradient_audit_checks_interface_components_and_resolves_endpoints(tmp_path, arm):
    from types import SimpleNamespace
    from verify_gradients import verify
    matrix=jnp.array([[.2,.1,-.1],[.1,-.2,.3]])
    residual=lambda y,x:y-matrix@x
    solver=LinearRoot(residual,options=Options(max_dofs=4,atol=1e-13))
    x=np.array([.2,.1,.3]);y=solver.solve(np.zeros(2),x)
    certificates=[]
    def certify(root,point):
        assert np.linalg.norm(np.asarray(residual(root,point))) < 1e-13
        certificates.append((root.copy(),point.copy()))
    rows=lambda y,x:jnp.array([jnp.vdot(y,y)+jnp.vdot(x,x),.3+y[0],1+y[1],.3+.1*x[1]])
    def interface_rows(y,x):
        vc=_manufactured_interface(y)
        field=lambda xyz:jnp.stack([1+.1*x[0]+.02*xyz[...,0],
            .03*x[1]+jnp.zeros(xyz.shape[:-1]),.02*x[2]+jnp.zeros(xyz.shape[:-1])],axis=-1)
        bn,jump,w=entry.boundary_residuals(vc,field)
        return jnp.array([entry.normal_cost(bn,w),entry.pressure_balance_cost(jump,w)])
    campaign=SimpleNamespace(arm=arm,x=x,y=y,rows=rows,interface_rows=interface_rows,out=tmp_path,
        model=SimpleNamespace(linear=solver,nb=1 if arm=='fixed' else 0,certify=certify))
    verify(campaign)
    report=json.loads((tmp_path/'gradient_verification.json').read_text())
    assert report['passed'] and report['optimizer_steps']==0
    assert len(certificates)==(12 if arm=='fixed' else 4)
    assert len(report['row_names'])==(6 if arm=='fixed' else 4)
    if arm=='fixed':
        assert report['row_names'][-2:]==['normal_field_loss','pressure_balance_loss']
    np.testing.assert_array_equal(campaign.x,x)
    np.testing.assert_array_equal(campaign.y,y)
