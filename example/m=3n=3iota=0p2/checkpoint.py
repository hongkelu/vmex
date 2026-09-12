"""Immutable accepted-state snapshots; legacy import requires an exact hash."""
import json
from pathlib import Path
import numpy as np
from vmex.core.solver import SpectralState
from case import sha256, PARAMETER_SCALES

STATE_NAMES = ('R_cos','R_sin','Z_cos','Z_sin','L_cos','L_sin')
LEGACY_SHA = 'c98d2e71d15f52d425c3e6644d67062a7a83df1debc4706e1085ce30a5b45a7e'
SCHEMA = 'vmex.m3n3-main-accepted/v1'


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def load(path, expected_sha, targets):
    if sha256(path) != expected_sha:
        raise ValueError('checkpoint hash mismatch')
    with np.load(path,allow_pickle=False) as f:
        data = {key:np.asarray(f[key]) for key in f.files}
    schema = str(data['schema_version'])
    legacy = schema == 'vmex.rotating_ellipse_fixed_qs_campaign/v5'
    if legacy and expected_sha != LEGACY_SHA:
        raise ValueError('only the authenticated strict step-10 legacy checkpoint is supported')
    if not legacy and schema != SCHEMA:
        raise ValueError('unsupported checkpoint schema')
    if not np.array_equal(data['parameter_scales'],PARAMETER_SCALES) or not np.array_equal(data['targets'],targets):
        raise ValueError('checkpoint targets/scales mismatch')
    p = data['parameters']
    if p.shape != (111,) or p.dtype != np.float64 or not np.all(np.isfinite(p)):
        raise ValueError('bad checkpoint parameter chart')
    step_array = data['accepted_step']
    if step_array.shape != () or not np.issubdtype(step_array.dtype,np.integer) or int(step_array) < 0:
        raise ValueError('bad absolute accepted step')
    state = SpectralState(*(data[n] for n in STATE_NAMES))
    return dict(data=data,state=state,parameters=p,step=int(step_array),legacy=legacy)


def save(output, accepted, step, targets, trust_radius, provenance):
    path = Path(output)/f'checkpoint_step_{step:04d}.npz'
    data = dict(schema_version=np.asarray(SCHEMA), parameters=np.asarray(accepted.parameters),
        parameter_scales=PARAMETER_SCALES, targets=targets, accepted_step=np.asarray(step),
        trust_radius=np.asarray(trust_radius),rcon0=np.asarray(accepted.rcon0),zcon0=np.asarray(accepted.zcon0),
        provenance_json=np.asarray(json.dumps(provenance,sort_keys=True)))
    data.update({n:np.asarray(getattr(accepted.state,n)) for n in STATE_NAMES})
    data.update({'mask_'+n:np.asarray(getattr(accepted.dof_mask,n)) for n in STATE_NAMES})
    with path.open('xb') as f:
        np.savez_compressed(f,**data)
    record = dict(path=str(path),sha256=sha256(path),accepted_step=step)
    write_json(Path(output)/'latest_checkpoint.json',record)
    return record
