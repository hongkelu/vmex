"""Selection must survive warmup bias, timing drift, and numerical failures."""
import dataclasses
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def tuner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module('adjoint_batch').tune_adjoint_batch


class Root:
    def __init__(self,jac=None,cfg=None):
        self.field_jacobian=np.asarray([[1.,2.],[0.,0.]]) if jac is None else jac
        self.closed=False
        self.cfg=cfg

    def close(self):
        self.closed=True


def run_probe(tuner,times,alter=None,deadline=1000):
    roots=[];events=[];now=[0.];calls=[]
    times={b:iter(t) for b,t in times.items()}
    def evaluate(batch):
        calls.append(batch)
        now[0]+=next(times[batch])
        root=Root();roots.append(root)
        if alter:
            alter(root,len(roots))
        return root
    try:
        result=tuner(evaluate,deadline=deadline,event=lambda phase,**kw:events.append((phase,kw)),clock=lambda:now[0])
        return result,roots,calls,events
    except BaseException:
        assert all(r.closed for r in roots)
        raise


def test_warmup_excluded_and_consistent_gain_selected(tuner):
    (batch,winner,report),roots,calls,events=run_probe(tuner,{32:[100,10,10,10],64:[200,8,8,8]})
    assert batch==64 and report['warm_median_seconds']=={'32':10,'64':8}
    assert calls==[32,64,32,64,64,32,32,64]
    assert all(r.closed == (r is not winner) for r in roots)
    assert events[-1][0]=='adjoint_batch_selected'


@pytest.mark.parametrize('t32,t64',[
    ([100,10,7.3,7.3],[200,7.5,7.1,7.4]),  # observed settling/drift
    ([100,10,10,10],[200,9.8,9.8,9.8]),     # improvement below 5%
    ([100,10,10,10],[200,8,8,10.5]),        # inconsistent win
])
def test_noisy_or_small_gain_keeps_32(tuner,t32,t64):
    (batch,_,report),_,_,_=run_probe(tuner,{32:t32,64:t64})
    assert batch==32 and report['reason']=='no_consistent_gain'


@pytest.mark.parametrize('change',[
    lambda x:x.__setitem__((0,0),1.001),
    lambda x:x.__setitem__((1,0),1e-20),
    lambda x:x.__setitem__((0,0),np.nan),
])
def test_bad_gradient_fails_closed_and_closes_every_object(tuner,change):
    def alter(root,count):
        if count==2:
            change(root.field_jacobian)
    with pytest.raises(ValueError,match='gradient'):
        run_probe(tuner,{32:[1]*4,64:[1]*4},alter)


def test_deadline_closes_all_roots(tuner):
    with pytest.raises(TimeoutError,match='batch tuning'):
        run_probe(tuner,{32:[1]*4,64:[1]*4},deadline=1.5)


def test_evaluation_failure_closes_prior_roots(tuner):
    root=Root()
    def evaluate(batch):
        if batch==64:
            raise RuntimeError('certification failed')
        return root
    with pytest.raises(RuntimeError,match='certification'):
        tuner(evaluate,deadline=float('inf'),event=lambda *a,**k:None)
    assert root.closed


def test_native_run_tunes_once_and_retains_matching_config(tuner,monkeypatch,tmp_path):
    import jax
    from vmex.core import freeboundary_continuation as fc
    support=importlib.import_module('single_stage_support')
    @dataclasses.dataclass(frozen=True)
    class Solver:
        adjoint_dense_batch_size:int=32
    @dataclasses.dataclass(frozen=True)
    class Config:
        solver:Solver
    configs=[];roots=[]
    def linearize(accepted,cfg,rhs,diagnostics,return_linearization):
        assert return_linearization
        configs.append(cfg);root=Root(cfg=cfg);roots.append(root)
        diagnostics.append(dict(accepted=True,relative_residual=0.))
        return root
    monkeypatch.setattr(fc,'free_boundary_continuation_state_pullback',linearize)
    monkeypatch.setattr(jax,'jacrev',lambda rows:lambda state:None)
    run=support.FreeBoundaryRun.__new__(support.FreeBoundaryRun)
    run.deadline=float('inf');run.linearization=None;run.point=None;run.step=0
    run.started=support.time.monotonic();run.output=tmp_path
    run.cfg=Config(Solver());initial_cfg=run.cfg;run.solver=run.cfg.solver
    run.batch_tuning_done=False;run.adjoint_batch_size=32
    run.accepted=SimpleNamespace(state=None);run.rows=None;run.values=np.ones(2);run.provenance={}
    run._tune_adjoint_batch([])
    assert len(configs)==8 and run.batch_tuning_done
    assert configs[0] is initial_cfg  # preserve the already compiled solver
    assert run.linearization.cfg is run.cfg and not run.linearization.closed
    assert run.provenance['adjoint_dense_batch_size']==run.adjoint_batch_size
    assert (tmp_path/'adjoint_batch_tuning.json').exists()
    previous=run.linearization
    # Restore the anchor while retaining the selected, already compiled solver.
    run.cfg=dataclasses.replace(initial_cfg,solver=run.solver)
    run._close_linearization()
    run.linearization = run._state_linearization(run.cfg, [])
    assert len(configs)==9 and previous.closed
    assert run.linearization.cfg is run.cfg
    assert run.cfg.solver.adjoint_dense_batch_size==run.adjoint_batch_size
    assert all(r.closed for r in roots[:-1])
    run._close_linearization()
    assert all(r.closed for r in roots)
