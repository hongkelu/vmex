import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

CASE = Path(__file__).resolve().parents[1]


def local_module(name):
    spec = importlib.util.spec_from_file_location('finitebeta_' + name, CASE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_reference_profiles_through_vmex_parser():
    from vmex.core.input import VmecInput
    from vmex.core import profiles
    prepare = local_module('prepare_inputs')
    prepared = VmecInput.from_json_text((CASE / 'inputs/input.json').read_text())
    reference = VmecInput.from_file(prepare.REFERENCE)
    assert prepared.curtor == reference.curtor == -6133652.330056256
    assert prepared.pres_scale == reference.pres_scale == 1.0
    assert prepared.pcurr_type == reference.pcurr_type == 'cubic_spline_ip'
    s = jnp.linspace(0., 1., 257)
    def p(inp):
        return profiles.pressure(inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f,
                                 s, pres_scale=inp.pres_scale, bloat=inp.bloat)
    def current(inp):
        return profiles.current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, s)
    np.testing.assert_array_equal(p(prepared), p(reference))
    np.testing.assert_array_equal(current(prepared), current(reference))
    assert float(p(prepared)[0]) == 720979.4853000008


def test_only_selected_input_quantities_change():
    prepare = local_module('prepare_inputs')
    inp, coils, metadata = prepare.build_inputs()
    parent = json.loads((prepare.PARENT / 'inputs/input.json').read_text())
    parent_coils = json.loads((prepare.PARENT / 'inputs/coils.json').read_text())
    allowed = set(prepare.reference_profiles(prepare.REFERENCE.read_text())) | {'extcur', 'phiedge'}
    assert all(value == inp[key] for key, value in parent.items() if key not in allowed)
    assert all(value == coils[key] for key, value in parent_coils.items() if key != 'dofs_currents')
    factor = metadata['signed_coil_current_and_flux_factor']
    np.testing.assert_allclose(np.array(coils['dofs_currents']) / parent_coils['dofs_currents'], factor, rtol=1e-15)
    assert inp['phiedge'] == pytest.approx(0.728301974551579)
    assert (CASE / 'inputs/original_vacuum_seed.npz').read_bytes() == (prepare.PARENT / 'inputs/seed_checkpoint.npz').read_bytes()
    assert json.loads((CASE / 'inputs/input.json').read_text()) == inp
    assert json.loads((CASE / 'inputs/coils.json').read_text()) == coils


def test_scaled_vacuum_state_b0_and_geometry():
    from vmex.core.input import VmecInput
    from vmex.core.solver import prepare_runtime, resolution_from_input, SpectralState
    from vmex.core import statephysics
    prepare = local_module('prepare_inputs')
    parent = VmecInput.from_json_text((prepare.PARENT / 'inputs/input.json').read_text())
    # Only field/flux scaling: no finite-beta equilibrium is claimed by this test.
    scaled_vacuum = dataclasses.replace(parent, phiedge=0.728301974551579)
    with np.load(CASE / 'inputs/vacuum_warm_start.npz', allow_pickle=False) as data:
        state = SpectralState(**{n: jnp.asarray(data[n]) for n in
            ('R_cos', 'R_sin', 'Z_cos', 'Z_sin', 'L_cos', 'L_sin')})
    with np.load(CASE / 'inputs/original_vacuum_seed.npz', allow_pickle=False) as data:
        old_state = SpectralState(**{n: jnp.asarray(data[n]) for n in
            ('R_cos', 'R_sin', 'Z_cos', 'Z_sin', 'L_cos', 'L_sin')})
    for name in ('R_cos','R_sin','Z_cos','Z_sin'):
        np.testing.assert_array_equal(getattr(state,name),getattr(old_state,name))
    for name in ('L_cos','L_sin'):
        np.testing.assert_array_equal(getattr(state,name),-getattr(old_state,name))
    old_rt = prepare_runtime(parent, resolution_from_input(parent))
    new_rt = prepare_runtime(scaled_vacuum, resolution_from_input(scaled_vacuum))
    old_b0 = float(statephysics.on_axis_magnetic_field(old_state, old_rt))
    new_b0 = float(statephysics.on_axis_magnetic_field(state, new_rt))
    assert old_b0 == pytest.approx(prepare.ORIGINAL_B0_T, rel=1e-12)
    assert new_b0 == pytest.approx(5.1, rel=1e-12)
    assert float(statephysics.mean_iota(state, new_rt)) == pytest.approx(float(statephysics.mean_iota(old_state, old_rt)), rel=1e-12)
    assert float(statephysics.aspect_ratio(state, new_rt)) == pytest.approx(float(statephysics.aspect_ratio(old_state, old_rt)), rel=1e-12)


def test_explicit_targets_do_not_follow_finitebeta_start():
    case = local_module('case')
    assert local_module('initialization').verify_runtime().is_dir()
    inp, builder, seed, contract, hashes = case.load_case()
    assert inp.phiedge > 0 and inp.curtor < 0 and inp.pres_scale > 0
    np.testing.assert_array_equal(case.resolve_targets([.8, 6., 6.5]), [.2, 5., 5.1])
    contract = json.loads((CASE / 'case_contract.json').read_text())
    assert contract['targets'] == {'mean_iota': .2, 'aspect_ratio': 5., 'b0': 5.1}
    assert contract['initial_equilibrium_solves'] == 1


@pytest.mark.parametrize('b0', [5.1, -0.17506474574437714])
def test_resume_requires_explicit_positive_b0_target(tmp_path, b0):
    import hashlib
    checkpoint = local_module('checkpoint')
    case = local_module('case')
    data = dict(schema_version=np.asarray('vmex.iota02-aspect5-finitebeta-accepted/v1'),
                accepted_step=np.asarray(0), parameters=np.zeros(111),
                parameter_scales=case.PARAMETER_SCALES,
                constraint_scales=case.CONSTRAINT_SCALES,
                targets=np.asarray([.2, 5., b0]), loss_scale=np.asarray(1.),
                phiedge=np.asarray(.728301974551579),
                rcon0=np.zeros((31,7,10)), zcon0=np.zeros((31,7,10)),
                contract_sha256=np.asarray('test-contract'),
                provenance_json=np.asarray(json.dumps({'input_hashes':{'input':'test-input'}})))
    for name in checkpoint.STATE_NAMES:
        data[name]=np.zeros((31,18))
        data['mask_'+name]=np.ones((31,18))
    path=tmp_path/'checkpoint.npz'
    np.savez_compressed(path,**data)
    kwargs=dict(contract_sha256='test-contract',input_hashes={'input':'test-input'},
                parameter_scales=case.PARAMETER_SCALES,constraint_scales=case.CONSTRAINT_SCALES,
                phiedge=.728301974551579,target_step=10)
    if b0 == 5.1:
        loaded=checkpoint.load_checkpoint(path,hashlib.sha256(path.read_bytes()).hexdigest(),**kwargs)
        assert loaded['targets'][2] == 5.1
    else:
        with pytest.raises(ValueError,match='targets mismatch'):
            checkpoint.load_checkpoint(path,hashlib.sha256(path.read_bytes()).hexdigest(),**kwargs)


@pytest.mark.parametrize('converged,edge,passes', [(False, 0., False), (True, 2e-10, False), (True, 5e-11, True)])
def test_initialization_requires_ordinary_force_gate(monkeypatch, tmp_path, converged, edge, passes):
    from vmex.core import freeboundary
    initialization = local_module('initialization')
    state = SimpleNamespace(**{n: np.zeros((3, 2)) for n in
        ('R_cos', 'R_sin', 'Z_cos', 'Z_sin', 'L_cos', 'L_sin')})
    result = SimpleNamespace(state=state, converged=converged, iterations=3,
                             fsqr=1e-12, fsqz=1e-12, fsql=1e-12, fedge=edge)
    stage = SimpleNamespace(result=result, rcon0=np.zeros(2), zcon0=np.zeros(2))
    calls = []
    def solve(inp, **kwargs):
        calls.append(kwargs)
        return stage
    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', solve)
    args = (SimpleNamespace(phiedge=.7283, pres_scale=1., am=[720979.], curtor=-6133652.),
            lambda p: object(), np.zeros(111), state,
            SimpleNamespace(rcon0=np.zeros(2), zcon0=np.zeros(2)),
            SimpleNamespace(resolution=object()),
            {'force_tolerance':1e-10, 'edge_force_tolerance':1e-10, 'max_iterations':12},
            lambda *a, **kw: None, time.monotonic()+30, tmp_path)
    if passes:
        assert initialization.ordinary_initial_state(*args) is stage
    else:
        with pytest.raises(RuntimeError, match='force gate'):
            initialization.ordinary_initial_state(*args)
    assert len(calls) == 1 and calls[0]['jacobian_retries'] == 0
    assert calls[0]['initial_state'] is state
    with np.load(tmp_path / 'ordinary_initial_state_uncertified.npz') as archive:
        assert not bool(archive['eligible_for_resume'])
