"""Neither maintained entry point can select a slow predictor or small batch."""
from pathlib import Path
import importlib
from types import SimpleNamespace
import pytest


@pytest.fixture
def support(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module('single_stage_support')


def test_default_policy_and_legacy_entrypoint(support):
    args=support.parse_args(['--output-dir','unused'],target_step=1,device='cpu',max_wall_hours=1)
    assert args.equilibrium_predictor=='reused_dense' and args.adjoint_dense_batch_size==32
    legacy=importlib.import_module('run')
    current=importlib.import_module('single_stage_free_boundary_optimization')
    assert legacy.main is current.main
    assert current.ADJOINT_BATCH_SIZE==32


@pytest.mark.parametrize('option,value',[('--equilibrium-predictor','tangent'),
    ('--equilibrium-predictor','accepted'),('--adjoint-dense-batch-size','4')])
def test_old_slow_launch_options_are_rejected(support,option,value):
    with pytest.raises(SystemExit) as exc:
        support.parse_args(['--output-dir','unused',option,value],target_step=1,device='cpu',max_wall_hours=1)
    assert exc.value.code==2


def test_programmatic_caller_cannot_bypass_fast_policy(support):
    with pytest.raises(ValueError,match='requires reused_dense'):
        support.FreeBoundaryRun(SimpleNamespace(equilibrium_predictor='accepted',adjoint_dense_batch_size=32))


@pytest.mark.parametrize('batch',[32,64])
def test_explicit_fast_batch_skips_auto_selection(support,batch):
    args=support.parse_args(['--output-dir','unused','--adjoint-dense-batch-size',str(batch)],
                            target_step=1,device='cpu',max_wall_hours=1)
    assert args.adjoint_dense_batch_size==batch
