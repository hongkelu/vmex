import json
from pathlib import Path
import numpy as np
from case import load_case,resolve_targets,CONSTRAINT_SCALES,make_rows,physical_rows
from optimization import Policy,constraint_state,proposal,acceptance,motion_bounds
from gates import pressure_diagnostic

TARGETS=np.array([.2,5.,5.1,11.067033897730942])

def test_four_physical_rows_and_profiles():
    from vmex.core import implicit as im
    from vmex.core.input import VmecInput
    import jax.numpy as jnp
    inp,builder,state,contract,_=load_case()
    assert inp.lfreeb and inp.mgrid_file=='DIRECT_ESSOS_BIOT_SAVART'
    from vmex.core.freeboundary import free_boundary_resolution
    resolution=free_boundary_resolution(inp,builder(np.zeros(111)))
    assert (resolution.ntheta,resolution.ntheta3,resolution.nzeta,resolution.ns)==(48,25,40,50)
    assert (inp.mpol,inp.ntor)==(3,3)
    assert inp.ncurr==1 and inp.curtor==-6133652.330056256
    assert contract['delbsq_limit']==.01
    assert contract['force_tolerance']==1e-10 and contract['forward_solve_tolerance']==1e-13
    assert contract['root_residual_atol']==2e-6 and contract['adjoint_residual_rtol']==2e-5
    assert np.array_equal(resolve_targets(TARGETS),TARGETS)
    from vmex.core.wout import read_wout
    w=read_wout(Path(__file__).parents[1]/'inputs/wout_fixed.nc')
    assert abs(w.Rmajor_p-TARGETS[3])<1e-8
    assert abs(np.mean(w.iotas[1:])-.2)<1e-5

def test_major_radius_rejects_otherwise_improving_qa():
    policy=Policy();before=np.array([1.,0.,0.,0.,0.]);jac=np.zeros((5,111));jac[0,0]=1
    jac[1:,1:5]=np.eye(4)
    direction=proposal(before,jac,np.ones(111),TARGETS,CONSTRAINT_SCALES,policy)
    assert direction.mode=='qa'
    after=before.copy();after[0]=.9;after[-1]=1.001
    ok,why=acceptance(before,after,direction,1.,TARGETS,CONSTRAINT_SCALES,policy)
    assert not ok and why['reason']=='constraint_violation'
    after[-1]=.99
    assert acceptance(before,after,direction,1.,TARGETS,CONSTRAINT_SCALES,policy)[0]
    assert motion_bounds(direction.delta)[0]<=.001 and motion_bounds(direction.delta)[1]<=.01

def test_qa_direction_annihilates_all_four_constraint_rows():
    rng=np.random.default_rng(19);jac=rng.normal(size=(5,111));values=np.r_[1.,np.zeros(4)]
    d=proposal(values,jac,np.ones(111),TARGETS,CONSTRAINT_SCALES,Policy())
    np.testing.assert_allclose(jac[1:]@d.delta,0,atol=1e-12)
    assert jac[0]@d.delta<0

def test_one_percent_pressure_gate():
    assert pressure_diagnostic(.01,1.,.01)['passed']
    assert not pressure_diagnostic(.010001,1.,.01)['passed']


def test_selected_target_lineage_and_unchanged_profiles():
    from case import CASE, sha256
    inp, builder, state, contract, hashes = load_case()
    ref=json.loads((CASE/'inputs/input.fixed.reference.json').read_text())
    actual=json.loads((CASE/'inputs/input.json').read_text())
    changed={k for k in ref if ref[k]!=actual[k]}
    assert changed=={'lfreeb','mgrid_file','ns_array','ftol_array','niter_array'}
    q=json.loads((CASE/'inputs/stage2_qualification.json').read_text())
    assert q['qualified_for_free_boundary_trial'] and not q['free_boundary_equilibrium_qualified']
    assert q['frozen_nestor']['coil_sha256']==hashes['coils.json']
    assert q['frozen_nestor']['state_sha256']==hashes['initial_fixed_state.npz']
    assert q['frozen_nestor']['wout_sha256']==hashes['wout_fixed.nc']
    assert q['frozen_nestor']['delbsq']<contract['delbsq_limit']
    assert inp.phiedge==91.09295154228212
