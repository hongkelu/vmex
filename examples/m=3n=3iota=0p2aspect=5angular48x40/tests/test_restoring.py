import dataclasses
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

CASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('restoring_optimizer', CASE/'optimization.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
TARGETS = np.array([.2, 5., -.17506474574437714])
CS = np.array([.005, .05, .01])


def example(error=-.9999999, normal_qa_slope=1000.):
    j = np.zeros((4, 111))
    j[0, 3] = 100.
    j[0, 5] = normal_qa_slope
    j[1, 4] = j[2, 5] = j[3, 6] = 1000.
    return np.array([1., 0., error, 0.]), j, np.ones(111)


def test_curved_band_old_tangent_fails_but_combination_passes():
    values, jac, scales = example()
    policy = m.Policy()
    direction = m.proposal(values, jac, scales, TARGETS, CS, policy)
    def evaluate(delta, trial):
        after = values+jac@delta
        # A QA move bends the aspect surface outwards despite Jc*d_QA=0.
        after[2] -= 20.*delta[3]**2
        return object(), after
    result = m.backtrack(values, direction, TARGETS, CS, policy, evaluate, lambda x: None)
    assert result.candidate is not None and result.trials[0]['accepted']
    assert result.values[0] < values[0]
    assert abs(result.values[2]) < abs(values[2])
    old_delta = np.zeros(111);old_delta[3] = -.001
    old = dataclasses.replace(direction, delta=old_delta)
    for alpha in (1., .5, .25, .125, .0625, .03125):
        # The very smallest old direction may enter the band in this synthetic
        # model; the full-size old direction must reproduce the boundary issue.
        _, after = evaluate(alpha*old_delta, 1)
        if alpha == 1.:
            assert not m.acceptance(values, after, old, alpha, TARGETS, CS, policy)[0]


def test_weight_increases_toward_boundary_and_preserves_descent():
    weights=[]
    for error in (-.2, -.85, -.9999999):
        values, jac, scales = example(error, normal_qa_slope=0.)
        d=m.proposal(values,jac,scales,TARGETS,CS,m.Policy())
        weights.append(d.restoring_weight)
        assert d.objective_directional_derivative < 0
        assert d.predicted_constraint_change[1] > 0
    assert weights[0] < weights[1] < weights[2]


def test_adverse_normal_gradient_is_limited_instead_of_proposing_uphill():
    values,jac,scales=example(normal_qa_slope=1e8)
    d=m.proposal(values,jac,scales,TARGETS,CS,m.Policy())
    assert 0 < d.restoring_weight < 1e-6
    assert d.objective_directional_derivative < 0


def test_qa_size_does_not_collapse_as_target_error_vanishes():
    qa=[]
    for error in (-1e-3, -1e-10, 0.):
        values,jac,scales=example(error, normal_qa_slope=0.)
        d=m.proposal(values,jac,scales,TARGETS,CS,m.Policy())
        qa.append(abs(d.delta[3]))
    assert min(qa) > .000999


def test_infeasible_case_retains_restoration_priority():
    values,jac,scales=example(-2.)
    d=m.proposal(values,jac,scales,TARGETS,CS,m.Policy())
    assert d.mode=='restoration'
    after=values+jac@d.delta
    assert m.acceptance(values,after,d,1.,TARGETS,CS,m.Policy())[0]


def test_true_acceptance_keeps_original_bands_and_qa_requirement():
    values,jac,scales=example()
    d=m.proposal(values,jac,scales,TARGETS,CS,m.Policy())
    outside=values.copy();outside[0]-=.01;outside[2]=-1.0000000001
    assert not m.acceptance(values,outside,d,1.,TARGETS,CS,m.Policy())[0]
    uphill=values.copy();uphill[0]+=.01;uphill[2]=-.9
    assert not m.acceptance(values,uphill,d,1.,TARGETS,CS,m.Policy())[0]


def test_recorded_step204_jacobian_predicts_inward_qa_descent():
    with np.load(CASE/'tests/fixtures/baseline_step204_proposal.npz',allow_pickle=False) as z:
        values,jac=z['rows'],z['jacobian']
    scales=np.r_[np.full(3,.06),np.tile([.002,.002,.002,.0005,.0005,.002/9,.002/9,.002/16,.002/16],12)]
    d=m.proposal(values,jac,scales,TARGETS,CS,m.Policy())
    assert d.mode=='qa_restoring' and d.restoring_weight>0
    assert d.objective_directional_derivative<0
    assert d.predicted_constraint_change[1]*CS[1]>1e-7
    assert np.all(values[1:]*(jac[1:]@d.delta)<=1e-14)
    motion,current=m.motion_bounds(d.delta)
    assert motion<=.001*(1+1e-12) and current<=.01*(1+1e-12)


@pytest.mark.parametrize('field,value', [('restoration_fraction_interior',0.),
    ('restoration_fraction_boundary',float('nan')),('restoration_descent_fraction',1.)])
def test_invalid_weights_rejected(field,value):
    with pytest.raises(ValueError):
        m.Policy(**{field:value})
