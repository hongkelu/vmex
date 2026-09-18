#!/usr/bin/env python3
"""Explicitly convert a historical case checkpoint to the public problem format.

A hash-verified new-format reference supplies the requested numerical identity.
Original input files authenticate the old lineage. No equilibrium is solved;
resuming the converted file must still certify its exact state and masks.
"""
from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def convert(source, source_sha256, reference, reference_sha256, legacy_inputs, output):
    """Convert unchanged numerical arrays after checking both input identities."""
    from vmex import VmecInput
    from vmex.core import _freeboundary_checkpoint as storage
    from vmex.core.coil_parameters import CoilParameters
    from essos.coils import Coils

    source, reference, output = map(Path, (source, reference, output))
    for path, digest in ((source, source_sha256), (reference, reference_sha256)):
        if not digest or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('checkpoint SHA256 mismatch')
    with np.load(reference, allow_pickle=False) as z:
        identity_text = str(z['identity'])
    storage.read(reference, reference_sha256, identity_text)
    identity = json.loads(identity_text)
    with np.load(source, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    if str(data['schema_version']) != 'vmex.iota02-aspect5-accepted/v1':
        raise ValueError('unsupported historical schema')
    provenance = json.loads(str(data['provenance_json']))
    contract = provenance['contract']
    inputs = Path(legacy_inputs).resolve()
    for name, digest in provenance['input_hashes'].items():
        path = (inputs/name).resolve()
        if not path.is_relative_to(inputs) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('historical input hash mismatch')
    inp = VmecInput.from_file(inputs/'input.json')
    inp = replace(inp, ftol_array=np.array([contract['force_tolerance']]),
                  niter_array=np.array([contract['max_iterations']]))
    input_values = json.loads(json.dumps(asdict(inp), default=lambda x: np.asarray(x).tolist()))
    if input_values != identity['input']:
        raise ValueError('effective input differs from the reference')
    coils = Coils.from_json(str(inputs/'coils.json'))
    coils.n_segments = 75  # Historical case used 75, independent of JSON display resolution.
    chart = CoilParameters.from_coils(coils,
                                    current_dofs=identity['current_dofs'], max_coil_mode=identity['max_coil_mode'],
                                    scales=identity['scales'])
    for name in ('coefficients', 'currents', 'nfp', 'stellsym', 'n_segments'):
        if not np.array_equal(getattr(chart, name), identity[name]):
            raise ValueError('coil chart differs: ' + name)
    context = identity['context']
    if provenance['optimizer_policy'] != context['objectives']['optimizer']:
        raise ValueError('optimizer policy differs; convert only after an explicit policy migration')
    if context['objectives']['qa_surfaces'] != [.25,.5,.75,1.] or context['objectives']['helicity'] != [1,0]:
        raise ValueError('QA objective differs')
    pairs = [('ftol','force_tolerance'), ('edge_force_tolerance','edge_force_tolerance'),
             ('max_iterations','max_iterations'), ('adjoint_solver','adjoint_solver'),
             ('adjoint_tol','adjoint_tol'), ('adjoint_residual_rtol','adjoint_residual_rtol'),
             ('adjoint_fail','adjoint_fail'), ('adjoint_dense_max_dofs','adjoint_dense_max_dofs')]
    if any(identity['solver'][new] != contract[old] for new, old in pairs):
        raise ValueError('solver settings differ')
    if context['root_residual_atol'] != contract['root_residual_atol'] or context['max_continuation_steps'] != contract['max_continuation_steps'] or context['continuation_step'] != .1:
        raise ValueError('continuation settings differ')
    if contract['root_polishing'] or contract['same_coil_retries'] != 0:
        raise ValueError('historical correction policy differs')
    for name, expected in [('parameter_scales',identity['scales']), ('phiedge',inp.phiedge),
                           ('targets',[c['target'] for c in identity['constraints']]),
                           ('constraint_scales',[c['scale'] for c in identity['constraints']])]:
        if not np.array_equal(data[name],expected):
            raise ValueError('saved numerical settings differ: ' + name)
    if any(c['rtol'] != .01 or c['atol'] != 0 for c in identity['constraints']):
        raise ValueError('physical target bands differ')
    gradient = data.get('initial_projected_gradient_norm')
    if int(data['accepted_step']) > 0 and gradient is None:
        raise ValueError('historical checkpoint lacks its stopping reference')
    data.update(schema_version=np.asarray(storage.SCHEMA), identity=np.asarray(identity_text),
                initial_gradient_norm=np.asarray(json.dumps(None if gradient is None else float(gradient))),
                converted_from_sha256=np.asarray(source_sha256))
    # Serialize without changing the numerical arrays.
    import io
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **data)
    # storage.read also checks the complete serialized checkpoint after writing.
    with output.open('xb') as stream:
        stream.write(buffer.getvalue())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    try:
        storage.read(output, digest, identity_text)
    except Exception:
        output.unlink()  # this failed conversion created this file
        raise
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--legacy-inputs', type=Path, required=True)
    args = parser.parse_args()
    print(convert(args.source, args.source_sha256, args.reference, args.reference_sha256,
                  args.legacy_inputs, args.output))


if __name__ == '__main__':
    main()
