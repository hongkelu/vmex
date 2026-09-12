"""Strict example contracts; no optimization campaign in unit tests."""
import dataclasses
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from vmex.core import freeboundary_implicit as fbi
from vmex.core.errors import AdjointSolveError

CASE = Path(__file__).resolve().parents[1]/'example/m=3n=3iota=0p2'
spec=importlib.util.spec_from_file_location('m3n3_optimization',CASE/'optimization.py')
opt=importlib.util.module_from_spec(spec);spec.loader.exec_module(opt)


def test_physical_step_and_equalities_are_preserved():
    rng=np.random.default_rng(12)
    jac=rng.normal(size=(5,111));scales=np.r_[np.full(3,.06),np.full(108,.002)]
    delta=opt.proposal(np.array([.7,0,0,0,0]),jac,scales)
    np.testing.assert_allclose(np.max(np.linalg.norm(opt.displacement(delta),axis=-1)),.001,rtol=1e-12)
    np.testing.assert_allclose(jac[1:]@delta,0,atol=1e-14)
    assert jac[0]@delta < 0


def test_degenerate_equalities_stop():
    with pytest.raises(RuntimeError,match='rank deficient'):
        opt.proposal(np.ones(5),np.ones((5,111)),np.ones(111))


def test_explicit_adjoint_gate_rejects_default_slack(monkeypatch):
    cfg=SimpleNamespace(adjoint_tol=2e-5,adjoint_gcrot_m=3,adjoint_gcrot_k=1,adjoint_maxiter=10)
    monkeypatch.setattr(fbi,'_reverse_gcrot_core',lambda *a,**k:(jnp.ones(3),jnp.array(1e-4),jnp.array(1.),jnp.array(2)))
    fbi._solve_prepared_reverse_adjoint(None,None,cfg) # unchanged default 2e-4
    diagnostics=[]
    with pytest.raises(AdjointSolveError):
        fbi._solve_prepared_reverse_adjoint(None,None,cfg,residual_rtol=2e-5,diagnostics=diagnostics)
    assert diagnostics[0]['accepted'] is False
    assert diagnostics[0]['tolerance']==2e-5


@pytest.mark.parametrize('nonlinear',[0.,.4])
def test_tangent_is_linear_response_without_equilibrium_solve(monkeypatch,nonlinear):
    from contextlib import nullcontext
    matrix=jnp.array([[2.,.3],[.1,3.]])
    coupling=jnp.array([[1.,2.],[.3,-1.]])
    warmed=[]
    def warm(params,cfg):
        warmed.append(cfg)
    def residual(z,p,f,base,r,zcon):
        assert warmed, 'runtime must be prepared before tracing'
        return matrix@z+nonlinear*z*z-coupling@f
    monkeypatch.setattr(fbi.im,'runtime_from_params',warm)
    monkeypatch.setattr(fbi.im,'_device_context',lambda *a:nullcontext())
    monkeypatch.setattr(fbi.im,'_device_pin',lambda cfg,values:values)
    monkeypatch.setattr(fbi,'_projected_residual',lambda *a:residual)
    monkeypatch.setattr(fbi.im,'_dof_projector',lambda *a:lambda x:x)
    monkeypatch.setattr(fbi,'_solve_free_boundary_stage',lambda *a,**k:pytest.fail('equilibrium solve'))
    cfg=SimpleNamespace(implicit=SimpleNamespace(adjoint_tol=1e-10,adjoint_gcrot_m=2,
        adjoint_gcrot_k=1,adjoint_maxiter=10),adjoint_residual_rtol=1e-10)
    direction=jnp.array([.2,-.1]); state=jnp.array([.1,-.2])
    field=jnp.linalg.solve(coupling,matrix@state+nonlinear*state*state)
    with jax.disable_jit(False),jax.checking_leaks():
        tangent=fbi.free_boundary_state_tangent(None,field,cfg,state,jnp.ones(2),direction,rcon0=None,zcon0=None)
    assert warmed == [cfg.implicit]
    expected=np.linalg.solve(matrix+jnp.diag(2*nonlinear*state),coupling@direction)
    np.testing.assert_allclose(tangent,expected,rtol=1e-9,atol=1e-12)


@pytest.mark.parametrize("edge",[1.1e-14,-1.,float('nan'),float('inf')])
def test_strict_edge_gate_rejects_without_changing_default(edge):
    from vmex.core.solver import _force_convergence
    args=(jnp.array(1e-16),jnp.array(1e-16),jnp.array(1e-16),jnp.array(edge),1e-14)
    assert bool(_force_convergence(*args))
    assert not bool(_force_convergence(*args,edge_tolerance=1e-14))


def test_strict_edge_gate_requires_vacuum_and_interior_channels():
    from vmex.core.solver import _force_convergence
    for i in range(3):
        channels=[1e-16]*3;channels[i]=2e-14
        assert not bool(_force_convergence(*channels,jnp.array(1e-16),1e-14,edge_tolerance=1e-14))
    assert not bool(_force_convergence(0.,0.,0.,jnp.array(0.),1e-14,edge_tolerance=1e-14,vacuum_active=False))
    assert bool(_force_convergence(1e-14,1e-14,1e-14,jnp.array(1e-14),1e-14,edge_tolerance=1e-14))
