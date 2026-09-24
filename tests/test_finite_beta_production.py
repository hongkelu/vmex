"""Finite-beta production wiring and physics contracts; no scientific campaign."""
import ast
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / 'examples/three-methods-finite-beta-benchmark'
spec = importlib.util.spec_from_file_location('finite_beta_production', HERE / 'finite_beta_production.py')
entry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = entry
spec.loader.exec_module(entry)
spec = importlib.util.spec_from_file_location('verify_finite_beta_production', HERE / 'verify_finite_beta_production.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def test_production_defaults_match_finite_beta_contract(tmp_path, capsys):
    args = entry.parse_args(['--device', 'cpu'])
    original = __import__('vmex').VmecInput.from_file(entry.INPUT)
    inp, _ = entry.load_input(args)
    assert inp.pres_scale == original.pres_scale and inp.phiedge == original.phiedge
    np.testing.assert_array_equal(inp.am, original.am)
    assert inp.ncurr == 1 and inp.curtor == 0 and not np.any(inp.ac)
    assert (inp.mpol, inp.ntor, inp.ns_array[-1]) == (8, 8, 51)
    assert args.ftol == 1e-15 and entry.ROOT_POLISH_TOLERANCE == 1e-12
    assert entry.BETA_REFERENCE == .005
    assert entry.INITIALIZATION_SECONDS == entry.OPTIMIZATION_SECONDS == 0
    output = tmp_path / 'dry-run'
    assert entry.main(['--dry-run', '--output', str(output)]) == 0
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)['optimizer'] == 'SLSQP'
    config = json.loads((HERE / 'benchmark.json').read_text())
    assert entry.CURVATURE_OBJECTIVE_LIMIT == config['coil_penalties']['curvature_hinge_per_m']
    assert entry.NORMAL_FIELD_WEIGHT == config['coil_penalties']['normal_field_weight']


@pytest.mark.parametrize('changes', [dict(curtor=1.), dict(ncurr=0), dict(ac=np.ones(21)),
    dict(pcurr_type='cubic_spline'), dict(pres_scale=0.), dict(pres_scale=float('nan')),
    dict(am=np.ones(21)), dict(lasym=True)])
def test_wrong_physics_rejected_before_solving(monkeypatch, changes):
    import vmex as vj
    inp = replace(vj.VmecInput.from_file(entry.INPUT), **changes)
    monkeypatch.setattr(vj.VmecInput, 'from_file', lambda path: inp)
    with pytest.raises(ValueError):
        entry.load_input(entry.parse_args([]))


def test_interface_uses_live_plasma_and_total_field_normalization(monkeypatch):
    """A manufactured plasma field proves it is neither omitted nor frozen."""
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import virtual_casing as vc
    import essos.fields
    calls = []
    def data(inp, state, **kw):
        calls.append(kw)
        return SimpleNamespace(B_total=jnp.ones((3, 2, 2)), state=state)
    def interface(data, **kwargs):
        assert kwargs['precision'] == 'fixed plan'
        # Coil field (1,2,0) plus state-dependent plasma field (s,0,0).
        total = jnp.broadcast_to(jnp.array([1.+data.state, 2., 0.])[:, None, None], (3, 2, 2))
        def field_check(external):
            np.testing.assert_allclose(external(jnp.ones((2, 2, 3)))[0, 0], [1., 2., 0.])
            return total
        return SimpleNamespace(weights=jnp.full((2, 2), .25), total_B_out=field_check,
            bnormal_residual=lambda field: jnp.full((2, 2), 1.+data.state),
            pressure_balance_residual=lambda field: jnp.full((2, 2), .03))
    monkeypatch.setattr(vj, 'surface_field_data_from_state', data)
    monkeypatch.setattr(vc, 'plan_vc_precision', lambda *a, **k: 'fixed plan')
    monkeypatch.setattr(vj.PlasmaVacuumInterface, 'from_surface_data', interface)
    monkeypatch.setattr(essos.fields, 'BiotSavart', lambda coils: SimpleNamespace(B=lambda xyz: jnp.array([1., 2., 0.])))
    monkeypatch.setattr(opt, 'volume_average_beta', lambda state, runtime: .005 + .001*state)
    rows, metrics = entry.make_interface_functions(None, SimpleNamespace(state=0., runtime=None))
    value = lambda state: jnp.sum(rows(state, None, None)[0])/4
    np.testing.assert_allclose(value(0.), 1/np.sqrt(5))
    np.testing.assert_allclose(value(1.), 1/np.sqrt(2))
    np.testing.assert_allclose(jax.grad(value)(1.), 4/8**1.5)
    report = metrics(1., None, None, 61, 64)
    assert report['beta_percent'] == .6
    np.testing.assert_allclose(report['pressure_jump_rms'], .01)
    assert calls[-1]['nphi'] == 61


def test_production_never_invokes_gradient_verifier(tmp_path, monkeypatch):
    from vmex import optimize as opt
    calls = []
    eq = SimpleNamespace(state=None, runtime=None,
        result=SimpleNamespace(fsqr=1e-16, fsqz=1e-16, fsql=1e-16, fedge=1e-16))
    class Problem:
        x0 = np.zeros(2)
        accepted_step = 0
        accepted = SimpleNamespace(parameters=x0, root_residual_norm=1e-13)
        solver_info = {}
        def enable_matrix_free(self, *a, **kw):
            assert not a and 'parity_rtol' not in kw
            calls.append('linear solver')
        def equilibrium_from_x(self, x): return eq
        def constraint_values(self, x): return np.array([.2, 1.])
        def fun(self, x): return 1.
        def coils_from_x(self, x): return None
        def evaluate_trial(self, *a, **kw): pytest.fail('production ran FD endpoints')
        def save_checkpoint(self, path): return {}
        def close(self): calls.append('close')
    stage = SimpleNamespace(problem=Problem(), inp=SimpleNamespace(ntor=0),
        qs=SimpleNamespace(total_state=lambda *a: .1), inequalities=lambda x: np.ones(3),
        interface_metrics=lambda *a: dict(beta_percent=.51, normal_field_rms=.001, normal_field_max=.002))
    def build(*a, **kw): calls.append('build'); return stage
    def optimize(*a, **kw):
        calls.append('optimize')
        return SimpleNamespace(stop_reason='accepted_step_budget_reached', success=False, status=99, message='budget')
    def endpoint(stage, args, summary, *rest):
        calls.append('endpoint'); summary.update(inequalities_met=True)
    monkeypatch.setattr(entry, 'setup_run', lambda args: tmp_path)
    monkeypatch.setattr(entry, 'build_problem', build)
    monkeypatch.setattr(entry, 'run_optimizer', optimize)
    monkeypatch.setattr(entry, 'verify_endpoint', endpoint)
    monkeypatch.setattr(entry, 'postprocess', lambda *a: None)
    monkeypatch.setattr(entry.free.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(entry.free.signal, 'alarm', lambda *a: None)
    monkeypatch.setattr(opt, 'aspect_ratio', lambda *a: 5.)
    monkeypatch.setattr(opt, 'boundary_from_state', lambda *a: (np.ones((1, 1)),))
    monkeypatch.setattr(verify, 'verify_problem', lambda *a: pytest.fail('verifier called'))
    assert entry.main(['--output', str(tmp_path)]) == 1
    assert calls == ['build', 'linear solver', 'optimize', 'endpoint', 'close']
    report = json.loads((tmp_path / 'optimization_summary.json').read_text())
    assert report['derivative_qualified'] is False
    assert report['final']['beta_percent'] == .51
    assert not (tmp_path / 'gradient_check.json').exists()


def test_separate_verifier_independently_solves_endpoints(tmp_path):
    calls = []
    class Problem:
        x0 = np.zeros(3)
        accepted = object()
        def value_and_grad(self, x): return 0., np.array([1., 2., 3.])
        def constraint_jac(self, x): return np.array([[2., 3., 4.], [3., 4., 5.]])
        def evaluate_trial(self, delta, *, predict, ftol):
            assert predict is False and ftol == 1e-15
            calls.append(delta)
            return object(), np.r_[np.array([1., 2., 3.]) @ delta, self.constraint_jac(None) @ delta]
        def enable_matrix_free(self, delta, **kw): return dict(passed=True)
    stage = SimpleNamespace(problem=Problem(), chart=SimpleNamespace(scales=np.ones(3)),
        constraint_transform=np.eye(2), inequalities=lambda values: values)
    checks = verify.verify_problem(stage, tmp_path)
    assert len(calls) == 4 and checks['finite_difference']['passed']
    np.testing.assert_array_equal(calls[0], -calls[1])
    np.testing.assert_array_equal(calls[2], -calls[3])


def test_production_contains_no_qualification_calls():
    tree = ast.parse(Path(entry.__file__).read_text())
    calls = {getattr(node.func, 'attr', getattr(node.func, 'id', '')) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not calls.intersection({'verify_problem', 'qualify', 'evaluate_trial'})
    assert {'from_loss', 'minimize', 'enable_root_polishing', 'enable_matrix_free'} <= calls


@pytest.mark.parametrize('resuming', [False, True])
def test_builder_preserves_historical_scalar_objective_and_gradient(tmp_path, monkeypatch, resuming):
    """Compare the two real loss implementations at identical manufactured physics."""
    from vmex import optimize as opt
    import vmex as vj
    import benchmark as c
    from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
    from essos.surfaces import surfacerzfourier_from_boundary
    scope = dict(jax=jax, jnp=jnp, opt=opt, c=c,
        im=SimpleNamespace(runtime_from_params=lambda *a: None),
        surfacerzfourier_from_boundary=surfacerzfourier_from_boundary,
        loss_coil_separation=loss_coil_separation, loss_coil_surface_distance=loss_coil_surface_distance,
        optimizer_rows=lambda value, physical, fixed: jnp.r_[value, c.inequalities(physical)])
    exec(compile((Path(__file__).parent / "data/finite_beta_scalar_loss.py.txt").read_text(),
                 "historical_scalar_loss", "exec"), scope)
    Reference = type("Reference", (), {"objective_rows": scope["objective_rows"]})
    captured = {}
    original = vj.VmecInput.from_file(entry.INPUT)
    qs = SimpleNamespace(total_state=lambda s, rt: jnp.sum(jnp.asarray([s, 2*s])**2),
                         residuals_state=lambda s, rt: jnp.asarray([s, 2*s]))
    monkeypatch.setattr(opt, 'QuasisymmetryRatioResidual', lambda *a: qs)
    monkeypatch.setattr(opt, 'aspect_ratio', lambda s, rt: 5.+s*.01)
    monkeypatch.setattr(opt, 'min_abs_iota', lambda s, rt: .18+s*.001)
    monkeypatch.setattr(opt, 'boundary_from_state', lambda *a: (original.rbc, original.zbs, None, None))
    monkeypatch.setattr(opt, 'solve_equilibrium', lambda inp, **kw:
        SimpleNamespace(state=.2, runtime=None))
    def rows(state, runtime, coils):
        normal = jnp.array([[.002, .003], [.004, .005]]) + state*.001 + coils.curves.dofs[0, 0, 0]*.0001
        return normal, jnp.full((2, 2), .25), jnp.zeros((2, 2))
    monkeypatch.setattr(entry, 'make_interface_functions', lambda *a: (rows, lambda *a: {}))
    class Problem:
        accepted_step = 5 if resuming else 0
        @property
        def x0(self): return captured['options']['parameterization'].x0
        def coils_from_x(self, x): return captured['options']['parameterization'].coils_from_x(x)
        def enable_root_polishing(self, **kw): captured['polishing'] = kw
        def close(self): pass
    def factory(inp, loss, **kwargs):
        captured.update(inp=inp, loss=loss, options=kwargs)
        return Problem()
    monkeypatch.setattr(opt.FreeBoundaryProblem, 'from_loss', factory)
    args = entry.parse_args(['--device', 'cpu', '--output', str(tmp_path)])
    if resuming:
        args.resume_checkpoint = tmp_path / 'accepted_0005.npz'
        np.savez(args.resume_checkpoint, accepted_step=5,
            identity=json.dumps(dict(context=dict(objectives=entry.qualification_contract(args)))))
        args.checkpoint_sha256 = entry.sha(args.resume_checkpoint)
    stage = entry.build_problem(args)
    if resuming:
        assert captured['options']['checkpoint'] == args.resume_checkpoint
        assert captured['options']['checkpoint_sha256'] == args.checkpoint_sha256
        assert 'restart_from' not in captured['options']
        assert json.loads((tmp_path / 'resume.json').read_text())['derivative_qualified'] is False
    assert stage.chart.x0.size == 99
    assert captured['inp'].curtor == 0 and captured['inp'].pres_scale == original.pres_scale
    assert captured['inp'].phiedge == original.phiedge
    assert len(captured['options']['quantities']) == 2  # beta is not constrained
    assert captured['polishing'] == dict(tolerance=1e-12)
    assert captured['options']['solver_options']['ftol'] == 1e-15
    assert captured['options']['solver_options']['edge_force_tolerance'] == 1e-15
    np.testing.assert_array_equal(stage.chart.currents, stage.coils.dofs_currents_raw)
    old = Reference()
    old.args = SimpleNamespace(fixed_pressure=True)
    old.inp, old.qs, old.solver = original, qs, SimpleNamespace(implicit=None)
    old.params = lambda u: None
    old.boundary = lambda state: (original.rbc, original.zbs)
    old.coils = stage.chart.coils_from_x
    old.normal_rows = lambda state, u: rows(state, None, old.coils(u))[:2]
    monkeypatch.setattr(c, 'physical_values', lambda state, rt: jnp.array([.005, .18, 1.]))
    point = jnp.asarray(stage.chart.x0)
    new_loss = lambda u: captured['loss'](.2, None, stage.chart.coils_from_x(u))
    old_loss = lambda u: old.objective_rows(.2, u)[0]
    new_value, new_grad = jax.value_and_grad(new_loss)(point)
    old_value, old_grad = jax.value_and_grad(old_loss)(point)
    np.testing.assert_allclose(new_value, old_value, rtol=1e-13)
    np.testing.assert_allclose(new_grad, old_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('flags', [
    ['--resume-checkpoint', 'saved.npz'], ['--checkpoint-sha256', 'abc'],
    ['--resume-checkpoint', 'saved.npz', '--checkpoint-sha256', 'abc', '--initial-coils', 'coils.json']])
def test_resume_rejects_ambiguous_or_unauthenticated_cli(flags):
    with pytest.raises(SystemExit):
        entry.parse_args(flags)


def test_checkpoint_migration_requires_exact_reviewed_contract(tmp_path):
    import copy
    args = entry.parse_args(['--device', 'cpu'])
    current = entry.qualification_contract(args)
    saved = copy.deepcopy(current)
    saved['numerical_functions_sha256']['build_problem'] = 'c5c2163f73b273b260662360c585aebe1dec8a6fba7b1ebc44e6f48e243e23db'
    saved['numerical_functions_sha256'].pop('checkpoint_restart')
    args.resume_checkpoint = tmp_path / 'step5.npz'
    np.savez(args.resume_checkpoint, accepted_step=5, identity=json.dumps(dict(context=dict(objectives=saved))))
    args.checkpoint_sha256 = entry.sha(args.resume_checkpoint)
    restart = entry.checkpoint_restart(args, current)
    assert restart['accepted_step'] == 5 and restart['contract'] == saved
    for field in ('input_sha256', 'ftol', 'core_sha256', 'dependencies'):
        wrong = copy.deepcopy(current)
        wrong[field] = 'changed'
        with pytest.raises(ValueError, match='differs'):
            entry.checkpoint_restart(args, wrong)
    wrong = copy.deepcopy(current)
    wrong['numerical_functions_sha256']['build_problem'] = 'unreviewed'
    with pytest.raises(ValueError, match='not the reviewed'):
        entry.checkpoint_restart(args, wrong)
    args.checkpoint_sha256 = 'bad'
    with pytest.raises(ValueError, match='SHA256'):
        entry.checkpoint_restart(args, current)
