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
    if str(data['schema_version']) != 'vmex.iota02-aspect5-accepted/v1':
        raise ValueError('checkpoint schema mismatch')
    provenance = json.loads(str(data['provenance_json']))
    if str(data['contract_sha256']) != contract_sha256:
        if previous_contract_path is None or current_contract is None:
            raise ValueError('checkpoint contract mismatch')
        previous_bytes = Path(previous_contract_path).read_bytes()
        if hashlib.sha256(previous_bytes).hexdigest() != str(data['contract_sha256']):
            raise ValueError('previous contract SHA256 mismatch')
        previous = json.loads(previous_bytes)
        if provenance['contract'] != previous:
            raise ValueError('previous contract provenance mismatch')
        if migration_path is None:
            expected = dict(previous, force_tolerance=1e-10, edge_force_tolerance=1e-10)
            if (previous['force_tolerance'] != 1e-14 or previous['edge_force_tolerance'] != 1e-14
                    or current_contract != expected):
                raise ValueError('only the authorized force and edge tolerance change is allowed')
        else:
            migration = json.loads(Path(migration_path).read_text())
            if (migration['previous_contract_sha256'] != str(data['contract_sha256'])
                    or migration['current_contract_sha256'] != contract_sha256):
                raise ValueError('contract migration hash mismatch')
            changes = {k: dict(before=previous.get(k), after=current_contract.get(k))
                       for k in set(previous) | set(current_contract)
                       if previous.get(k) != current_contract.get(k)}
            if changes != migration['changes']:
                raise ValueError('unrecorded contract change')
            if set(changes) != {'proposal', 'nonlinear_constraint_acceptance', 'optimizer_policy_sha256'}:
                raise ValueError('migration must change only optimizer policy metadata')
    if provenance['input_hashes'] != input_hashes:
        raise ValueError('checkpoint input lineage mismatch')
    step = data['accepted_step']
    if step.shape != () or step.dtype.kind not in 'iu' or not 0 <= int(step) < target_step:
        raise ValueError('target must exceed the saved absolute step')
    shapes = {'parameters': (111,), 'parameter_scales': (111,), 'targets': (3,),
              'constraint_scales': (3,), 'loss_scale': (), 'phiedge': (),
              'rcon0': (31, 7, 10), 'zcon0': (31, 7, 10)}
    shapes.update({name: (31, 18) for name in STATE_NAMES})
    shapes.update({'mask_' + name: (31, 18) for name in STATE_NAMES})
    for name, shape in shapes.items():
        x = data[name]
        if x.shape != shape or x.dtype != np.float64 or not np.all(np.isfinite(x)):
            raise ValueError('invalid float64 checkpoint field: ' + name)
    for name, expected in [('parameter_scales', parameter_scales),
                           ('constraint_scales', constraint_scales), ('phiedge', phiedge)]:
        if not np.array_equal(data[name], expected):
            raise ValueError('checkpoint mismatch: ' + name)
    if not np.array_equal(data['targets'][:2], [.2, 5.]) or data['targets'][2] >= 0:
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
