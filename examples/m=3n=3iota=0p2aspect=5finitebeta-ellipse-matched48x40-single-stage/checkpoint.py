"""Authenticated accepted-state resume; targets and normalization stay frozen."""
import hashlib
import json
from pathlib import Path
import numpy as np

STATE_NAMES = ('R_cos', 'R_sin', 'Z_cos', 'Z_sin', 'L_cos', 'L_sin')


def load_checkpoint(path, expected_sha256, *, contract_sha256, input_hashes,
                    parameter_scales, constraint_scales, phiedge, target_step,
                    previous_contract_path=None, current_contract=None, migration_path=None):
    path = Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('checkpoint SHA256 mismatch')
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    if str(data['schema_version']) != 'vmex.finitebeta-ellipse-four-constraints/v1':
        raise ValueError('checkpoint schema mismatch')
    provenance = json.loads(str(data['provenance_json']))
    if str(data['contract_sha256']) != contract_sha256:
        raise ValueError('checkpoint contract mismatch; vacuum migrations are not supported')
    if provenance['input_hashes'] != input_hashes:
        raise ValueError('checkpoint input lineage mismatch')
    step = data['accepted_step']
    if step.shape != () or step.dtype.kind not in 'iu' or not 0 <= int(step) < target_step:
        raise ValueError('target must exceed the saved absolute step')
    if current_contract is None:
        raise ValueError('case contract required for angular-grid checkpoint validation')
    baseline_shape=(int(current_contract['ns']),int(current_contract['ntheta'])//2+1,int(current_contract['nzeta']))
    shapes = {'parameters': (111,), 'parameter_scales': (111,), 'targets': (4,),
              'constraint_scales': (4,), 'loss_scale': (), 'phiedge': (),
              'rcon0': baseline_shape, 'zcon0': baseline_shape}
    shapes.update({name: (50, 18) for name in STATE_NAMES})
    shapes.update({'mask_' + name: (50, 18) for name in STATE_NAMES})
    for name, shape in shapes.items():
        x = data[name]
        if x.shape != shape or x.dtype != np.float64 or not np.all(np.isfinite(x)):
            raise ValueError('invalid float64 checkpoint field: ' + name)
    for name, expected in [('parameter_scales', parameter_scales),
                           ('constraint_scales', constraint_scales), ('phiedge', phiedge)]:
        if not np.array_equal(data[name], expected):
            raise ValueError('checkpoint mismatch: ' + name)
    if not np.array_equal(data['targets'], [.2, 5., 5.1, 11.067033897730942]):
        raise ValueError('checkpoint targets mismatch')
    if float(data['loss_scale']) <= 0:
        raise ValueError('invalid QA normalization')
    for name in STATE_NAMES:
        if not np.all(np.isin(data['mask_' + name], [0., 1.])):
            raise ValueError('invalid saved mask: ' + name)
    return data


def verify_restoration(accepted, data):
    """Certification must evaluate the checkpoint, never replace its state."""
    pairs = [(accepted.parameters, data['parameters']),
             (accepted.rcon0, data['rcon0']), (accepted.zcon0, data['zcon0'])]
    pairs += [(getattr(accepted.state, n), data[n]) for n in STATE_NAMES]
    pairs += [(getattr(accepted.dof_mask, n), data['mask_' + n]) for n in STATE_NAMES]
    if any(not np.array_equal(np.asarray(a), b) for a, b in pairs):
        raise RuntimeError('checkpoint state, masks, parameters or baselines changed during certification')
