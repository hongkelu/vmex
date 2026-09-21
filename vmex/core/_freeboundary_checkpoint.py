"""Lossless accepted-root storage for the free-boundary problem API."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

FIELDS = ('R_cos', 'R_sin', 'Z_cos', 'Z_sin', 'L_cos', 'L_sin')
SCHEMA = 'vmex.freeboundary-problem/v1'


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False,
                      default=lambda x: np.asarray(x).tolist())


def identity(inp, chart, constraints, options, context):
    """Bind numerical inputs and caller-supplied objective/optimizer definitions."""
    options = {k: v for k, v in options.items() if k != 'device'}
    return _json(dict(input=asdict(inp), coefficients=chart.coefficients,
                      currents=chart.currents, current_dofs=chart.current_dofs,
                      scales=chart.scales, max_coil_mode=chart.mode,
                      nfp=chart.nfp, stellsym=chart.stellsym, n_segments=chart.n_segments,
                      constraints=[dict(target=c.target, scale=c.scale, rtol=c.rtol, atol=c.atol)
                                   for c in constraints], solver=options, context=context))


def read(path, expected_sha256, expected_identity):
    """Authenticate and validate a checkpoint before any solver is constructed."""
    path = Path(path)
    if not expected_sha256 or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('checkpoint SHA256 mismatch')
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    if str(data['schema_version']) != SCHEMA:
        raise ValueError('unsupported checkpoint schema; convert historical checkpoints explicitly')
    if str(data['identity']) != expected_identity:
        raise ValueError('checkpoint input, objective, optimizer or solver settings differ')
    step = data['accepted_step']
    if step.shape != () or step.dtype.kind not in 'iu' or int(step) < 0:
        raise ValueError('invalid accepted step')
    for name in ('parameters', 'rcon0', 'zcon0', 'loss_scale', *FIELDS,
                 *('mask_' + f for f in FIELDS)):
        value = data[name]
        if value.dtype != np.float64 or not np.all(np.isfinite(value)):
            raise ValueError('invalid float64 checkpoint field: ' + name)
    if data['parameters'].ndim != 1 or data['loss_scale'].shape != () or data['loss_scale'] <= 0:
        raise ValueError('invalid checkpoint parameters or normalization')
    for name in FIELDS:
        if data[name].ndim != 2 or data[name].shape != data[FIELDS[0]].shape:
            raise ValueError('invalid state shape: ' + name)
        mask = data['mask_' + name]
        if mask.shape != data[name].shape or not np.all(np.isin(mask, (0., 1.))):
            raise ValueError('invalid checkpoint mask: ' + name)
    if data['rcon0'].ndim != 3 or data['zcon0'].shape != data['rcon0'].shape:
        raise ValueError('invalid constraint baselines')
    reference = json.loads(str(data['initial_gradient_norm']))
    if reference is not None and (not np.isfinite(reference) or reference < 0):
        raise ValueError('invalid initial gradient reference')
    data['gradient_reference'] = reference
    return data


def verify(accepted, data):
    """Certification must preserve every saved numerical array exactly."""
    pairs = [(accepted.parameters, data['parameters']),
             (accepted.rcon0, data['rcon0']), (accepted.zcon0, data['zcon0'])]
    pairs += [(getattr(accepted.state, f), data[f]) for f in FIELDS]
    pairs += [(getattr(accepted.dof_mask, f), data['mask_' + f]) for f in FIELDS]
    if any(not np.array_equal(np.asarray(value), saved) for value, saved in pairs):
        raise ValueError('checkpoint state, masks, parameters or baselines changed during certification')


def write(problem, path):
    """Write only the accepted root; existing checkpoint files are preserved."""
    if problem.checkpoint_identity is None:
        raise ValueError('provide checkpoint_identity with objective and optimizer settings')
    accepted = problem.accepted
    data = dict(schema_version=np.asarray(SCHEMA), identity=np.asarray(problem.checkpoint_identity),
                accepted_step=np.asarray(problem.accepted_step),
                initial_gradient_norm=np.asarray(_json(problem.initial_gradient_norm)),
                parameters=np.asarray(accepted.parameters), loss_scale=np.asarray(problem.loss_scale),
                rcon0=np.asarray(accepted.rcon0), zcon0=np.asarray(accepted.zcon0))
    data.update({f: np.asarray(getattr(accepted.state, f)) for f in FIELDS})
    data.update({'mask_' + f: np.asarray(getattr(accepted.dof_mask, f)) for f in FIELDS})
    path = Path(path).resolve()
    with path.open('xb') as stream:
        np.savez_compressed(stream, **data)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                accepted_step=problem.accepted_step)
